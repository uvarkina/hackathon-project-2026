"""
Analysis module — AI-voice scoring.

Two engines, chosen automatically at runtime:

  1. Conformer model (conformer_audio_model.keras) — preferred.
     • Loaded WEIGHTS-ONLY into a code-rebuilt architecture, so it is
       immune to the Keras-version mismatch that crashed full-model loading
       (saved with Keras 3.14, '.keras' files won't deserialize on 3.12/3.13:
        "Unrecognized keyword arguments passed to Dense: {'quantization_config'}").
     • Training label convention was AI=0, Human=1, so the sigmoid output is
       P(human).  AI score is therefore (1 - output).  <-- critical inversion.

  2. Heuristic fallback (librosa MFCC/bandwidth) — used only if TensorFlow or
     the weights file are missing/broken, so the backend NEVER fails to start.

Returned dict: {"score": float 0..1 (1=AI), "speech": bool, "engine": str}
"""
import os
import threading
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import asyncio

TARGET_SR = 16000
N_MELS = 80
MAX_FRAMES = 400  # model input is fixed at 400 frames ≈ 4 s @ hop 160 / 16 kHz
MODEL_WINDOW_SEC = 4.0  # feed the model ~its full input length, not a bare 3 s chunk
SILENCE_RMS = 0.008  # below this the chunk is treated as silence (not flagged)
# "pretrained" | "model" | "heuristic" — override without code edits.
#   pretrained = HuggingFace wav2vec2 anti-spoof model (recommended sanity check)
#   model      = your local conformer_audio_model.keras
#   heuristic  = librosa MFCC/bandwidth fallback
VOICE_ENGINE = os.environ.get("VOICE_ENGINE", "model").strip().lower()
# HF anti-spoofing model. Labels: 0=fake (AI), 1=real (human).
SPOOF_MODEL_ID = os.environ.get("SPOOF_MODEL", "MelodyMachine/Deepfake-audio-detection-V2")
# Does the conformer's sigmoid output mean P(AI) or P(human)?
#   • Current shipped conformer_audio_model.keras -> output is P(human): keep 0.
#   • After retraining with the fixed retrain_model.py (0=human, 1=AI),
#     output becomes P(AI): set MODEL_OUTPUT_IS_AI=1 in the environment.
MODEL_OUTPUT_IS_AI = os.environ.get("MODEL_OUTPUT_IS_AI", "0").strip().lower() in ("1", "true", "yes")

_PT = None          # (feature_extractor, model, torch, fake_idx)
_PT_TRIED = False
_PT_LOCK = threading.Lock()

_MODEL = None
_MODEL_TRIED = False
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()  # serialise predict() across executor threads

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "conformer_audio_model.keras")


# ---------------------------------------------------------------------------
# Model: rebuild architecture in code, load weights only (version-proof)
# ---------------------------------------------------------------------------

def _build_and_load_model():
    """Rebuild the Conformer in code and load weights from the .keras zip.
    Raises on any failure — caller decides whether to fall back."""
    import zipfile
    import tempfile

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    class ConvModule(layers.Layer):
        def __init__(self, d, ks=31, **kw):
            super().__init__(**kw)
            self.norm = layers.LayerNormalization()
            self.pw1 = layers.Conv1D(d * 2, 1, padding="same")
            self.dw = layers.DepthwiseConv1D(ks, padding="same")
            self.bn = layers.BatchNormalization()
            self.pw2 = layers.Conv1D(d, 1, padding="same")
            self.drop = layers.Dropout(0.1)

        def call(self, x, training=False):
            r = x; x = self.norm(x); x = self.pw1(x)
            x, g = tf.split(x, 2, axis=-1); x = x * tf.sigmoid(g)
            x = self.dw(x); x = self.bn(x, training=training)
            x = tf.nn.silu(x); x = self.pw2(x)
            return r + self.drop(x, training=training)

    class FFModule(layers.Layer):
        def __init__(self, d, **kw):
            super().__init__(**kw)
            self.norm = layers.LayerNormalization()
            self.fc1 = layers.Dense(d * 4, activation="swish")
            self.drop = layers.Dropout(0.1)
            self.fc2 = layers.Dense(d)

        def call(self, x, training=False):
            r = x; x = self.norm(x); x = self.fc1(x)
            x = self.drop(x, training=training); x = self.fc2(x)
            return r + 0.5 * x

    class ConfBlock(layers.Layer):
        def __init__(self, d, h, ks=31, **kw):
            super().__init__(**kw)
            self.ff1 = FFModule(d); self.norm = layers.LayerNormalization()
            self.mhsa = layers.MultiHeadAttention(num_heads=h, key_dim=d // h, dropout=0.1)
            self.drop = layers.Dropout(0.1); self.conv = ConvModule(d, ks)
            self.ff2 = FFModule(d); self.ln = layers.LayerNormalization()

        def call(self, x, training=False):
            x = self.ff1(x, training=training)
            r = x; xn = self.norm(x)
            x = r + self.drop(self.mhsa(xn, xn, training=training), training=training)
            x = self.conv(x, training=training)
            x = self.ff2(x, training=training)
            return self.ln(x)

    def build_model(d=144, h=4, blocks=4, ks=31):
        inp = keras.Input(shape=(MAX_FRAMES, N_MELS))
        x = layers.Dense(d)(inp)
        pos = tf.cast(tf.range(MAX_FRAMES), tf.float32)[:, tf.newaxis]
        dims = tf.cast(tf.range(0, d, 2), tf.float32)[tf.newaxis, :]
        scale = tf.pow(10000.0, dims / tf.cast(d, tf.float32))
        pe = tf.reshape(tf.stack([tf.sin(pos / scale), tf.cos(pos / scale)], axis=-1), (MAX_FRAMES, d))
        x = x + pe[tf.newaxis]
        x = layers.Dropout(0.1)(x)
        for i in range(blocks):
            x = ConfBlock(d, h, ks, name=f"conf_{i}")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="swish")(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        return keras.Model(inputs=inp, outputs=out)

    model = build_model()
    with zipfile.ZipFile(_MODEL_PATH) as z:
        weights = z.read("model.weights.h5")
    tmp = os.path.join(tempfile.gettempdir(), "conformer_audio_model.weights.h5")
    with open(tmp, "wb") as f:
        f.write(weights)
    model.load_weights(tmp)
    # warm up so the first real request isn't slow / doesn't race
    model.predict(np.zeros((1, MAX_FRAMES, N_MELS), dtype=np.float32), verbose=0)
    return model


def _get_model():
    """Lazy, thread-safe, crash-proof model accessor. Returns model or None."""
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL_TRIED:
            return _MODEL
        try:
            _MODEL = _build_and_load_model()
            print("[analysis] Conformer model loaded (weights-only).")
        except Exception as e:
            _MODEL = None
            print(f"[analysis] Model unavailable -> heuristic fallback "
                  f"({type(e).__name__}: {e})")
        _MODEL_TRIED = True
    return _MODEL


# ---------------------------------------------------------------------------
# Pretrained HuggingFace anti-spoofing model (wav2vec2)
# ---------------------------------------------------------------------------

def _get_pretrained():
    """Lazy, thread-safe loader for the HF wav2vec2 spoof model. Returns tuple or None."""
    global _PT, _PT_TRIED
    if _PT_TRIED:
        return _PT
    with _PT_LOCK:
        if _PT_TRIED:
            return _PT
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            fe = AutoFeatureExtractor.from_pretrained(SPOOF_MODEL_ID)
            model = AutoModelForAudioClassification.from_pretrained(SPOOF_MODEL_ID)
            model.eval()
            # locate the "fake" class index from the model's own labels
            id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
            fake_idx = next((i for i, lbl in id2label.items()
                             if lbl in ("fake", "spoof", "ai", "synthetic")), 0)
            _PT = (fe, model, torch, fake_idx)
            print(f"[analysis] Pretrained spoof model loaded: {SPOOF_MODEL_ID} "
                  f"(fake_idx={fake_idx})")
        except Exception as e:
            _PT = None
            print(f"[analysis] Pretrained model unavailable "
                  f"({type(e).__name__}: {e})")
        _PT_TRIED = True
    return _PT


def _score_pretrained(wav, sr):
    """Return P(fake) in 0..1 using the wav2vec2 model, or None if unavailable."""
    import numpy as np
    bundle = _get_pretrained()
    if bundle is None:
        return None
    fe, model, torch, fake_idx = bundle
    try:
        import librosa
        if sr != TARGET_SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        inputs = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        with _INFER_LOCK, torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[fake_idx].item())
    except Exception as e:
        print(f"[analysis] pretrained inference failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def decode_audio(path: str):
    """Decode an audio file to (float32 mono waveform, sample_rate)."""
    import numpy as np
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        sr = int(sr)
    except Exception:
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=True)
        sr = int(sr)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def _rms(wav) -> float:
    import numpy as np
    if wav is None or wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))


def _extract_mel(wav, sr):
    """Waveform -> (1, MAX_FRAMES, N_MELS) log-mel, exactly as in training."""
    import numpy as np
    import librosa
    if sr != TARGET_SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    mel = librosa.feature.melspectrogram(
        y=wav, sr=TARGET_SR, n_fft=512, hop_length=160, win_length=400,
        n_mels=N_MELS, fmin=0, fmax=8000, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    std = log_mel.std()
    log_mel = (log_mel - log_mel.mean()) / (std if std > 1e-8 else 1.0)
    feat = log_mel.T.astype(np.float32)
    if feat.shape[0] >= MAX_FRAMES:
        feat = feat[:MAX_FRAMES]
    else:
        feat = np.concatenate(
            [feat, np.zeros((MAX_FRAMES - feat.shape[0], N_MELS), dtype=np.float32)])
    return feat[np.newaxis, ...]


def _score_heuristic(wav, sr) -> float:
    """Fallback: 70% MFCC variance + 30% bandwidth std (librosa only)."""
    import numpy as np
    import librosa
    if sr != TARGET_SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
    mfcc_score = max(0.0, min(1.0, (mfcc_var - 1500) / 1300.0))
    bw = librosa.feature.spectral_bandwidth(y=wav, sr=sr)[0]
    bw_std = float(np.std(bw))
    bw_score = max(0.0, min(1.0, (bw_std - 350) / 350.0))
    return round(mfcc_score * 0.70 + bw_score * 0.30, 4)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_sync(wav, sr) -> dict:
    """Blocking scorer — run inside an executor thread."""
    import numpy as np

    # Use only the most recent ~4 s (the model's full input length).
    win = int(sr * MODEL_WINDOW_SEC)
    if wav.size > win:
        wav = wav[-win:]

    rms = _rms(wav)
    if rms < SILENCE_RMS:
        # Silence / very quiet: do NOT flag. Garbage features here are the
        # main reason real, quiet speech used to read as "AI".
        return {"score": 0.0, "speech": False, "rms": round(rms, 5), "engine": "gate"}

    # 1) Pretrained wav2vec2 spoof model (P(fake) == AI score directly)
    if VOICE_ENGINE == "pretrained":
        ai = _score_pretrained(wav, sr)
        if ai is not None:
            return {"score": max(0.0, min(1.0, ai)), "speech": True,
                    "rms": round(rms, 5), "engine": "pretrained"}

    # 2) Local conformer model (output is P(human) -> invert)
    model = None if VOICE_ENGINE == "heuristic" else _get_model()
    if model is not None:
        try:
            feat = _extract_mel(wav, sr)
            with _INFER_LOCK:
                p = float(model.predict(feat, verbose=0).flatten()[0])
            # Current model outputs P(human) -> invert. Retrained model (env
            # MODEL_OUTPUT_IS_AI=1) outputs P(AI) directly -> use as-is.
            ai_score = p if MODEL_OUTPUT_IS_AI else (1.0 - p)
            return {"score": max(0.0, min(1.0, ai_score)), "speech": True,
                    "rms": round(rms, 5), "engine": "model"}
        except Exception as e:
            print(f"[analysis] model inference failed, heuristic this chunk: {e}")

    # 3) Heuristic fallback (never crashes the backend)
    return {"score": _score_heuristic(wav, sr), "speech": True,
            "rms": round(rms, 5), "engine": "heuristic"}


async def analyze_waveform(wav, sr) -> dict:
    """Async wrapper: score an already-decoded waveform (used with a rolling buffer)."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _score_sync, wav, sr)
    except Exception:
        return {"score": 0.0, "speech": False, "engine": "error"}


async def analyze_voice(audio_path: str) -> dict:
    """Async: decode one file and score it. Kept for backward compatibility."""
    try:
        wav, sr = decode_audio(audio_path)
    except Exception:
        return {"score": 0.0, "speech": False, "engine": "decode_error"}
    return await analyze_waveform(wav, sr)
