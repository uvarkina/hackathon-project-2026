"""
retrain_model.py
================
Retrain the Conformer model using:
  1. The HuggingFace English deepfake dataset (1,870 samples)
  2. Your local Hebrew/Russian files (augmented to ~500 samples)

This creates a model that works on BOTH English and Hebrew AI detection.
"""
import os, sys, warnings, logging
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import soundfile as sf
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET_SR = 16000
N_MELS = 80
MAX_FRAMES = 400

# ── Your labeled local files ──────────────────────────────────────────────────
LOCAL_FILES = [
    ("text_fraud_recognition/hebrew_3_ai.mp3", 0),         # AI = 0
    ("text_fraud_recognition/speach_hebrew_ai.mp3", 0),    # AI = 0
    ("text_fraud_recognition/speach_hebrew_1.mp3", 1),     # Human = 1
]
# All "28 мая" files are human
import glob
for f in sorted(glob.glob("text_fraud_recognition/28*.mp3")):
    LOCAL_FILES.append((f, 1))


def extract_mel(waveform, sr, max_frames=MAX_FRAMES):
    """Waveform → Log-Mel features (max_frames, N_MELS)."""
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    mel = librosa.feature.melspectrogram(y=waveform, sr=TARGET_SR, n_fft=512,
                                          hop_length=160, win_length=400, n_mels=N_MELS,
                                          fmin=0, fmax=8000, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    std = log_mel.std()
    log_mel = (log_mel - log_mel.mean()) / (std if std > 1e-8 else 1.0)
    feat = log_mel.T.astype(np.float32)
    if feat.shape[0] >= max_frames:
        feat = feat[:max_frames]
    else:
        feat = np.concatenate([feat, np.zeros((max_frames - feat.shape[0], N_MELS), dtype=np.float32)])
    return feat


def augment_waveform(waveform, sr):
    """Generate augmented versions: noise, pitch shift, speed change, volume."""
    augmented = []
    # Original
    augmented.append(waveform.copy())
    # Add noise at different levels
    for noise_level in [0.002, 0.005, 0.01, 0.02]:
        noisy = waveform + noise_level * np.random.randn(len(waveform)).astype(np.float32)
        augmented.append(noisy)
    # Pitch shift
    for n_steps in [-2, -1, 1, 2]:
        shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
        augmented.append(shifted)
    # Speed change
    for rate in [0.85, 0.9, 1.1, 1.15]:
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        augmented.append(stretched)
    # Volume change
    for gain in [0.5, 0.7, 1.3, 1.5]:
        augmented.append(waveform * gain)
    # Random crop (take different 4-second segments if long enough)
    seg_len = TARGET_SR * 4
    if len(waveform) > seg_len * 2:
        for _ in range(5):
            start = np.random.randint(0, len(waveform) - seg_len)
            augmented.append(waveform[start:start + seg_len])
    return augmented


def load_local_files():
    """Load and augment local labeled files."""
    X_list, y_list = [], []
    for path, label in LOCAL_FILES:
        if not os.path.isfile(path):
            log.warning("File not found: %s", path)
            continue
        try:
            w, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            try:
                w, sr = librosa.load(path, sr=None, mono=True, res_type="soxr_hq")
                sr = int(sr)
            except Exception:
                log.warning("Cannot load: %s", path)
                continue

        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != TARGET_SR:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")

        # Augment
        augmented = augment_waveform(w, TARGET_SR)
        for aug_wav in augmented:
            feat = extract_mel(aug_wav, TARGET_SR)
            X_list.append(feat)
            y_list.append(label)

    log.info("Local files: %d samples (from %d files)", len(X_list), len(LOCAL_FILES))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_hf_dataset():
    """Load the HuggingFace dataset from cached parquet files."""
    import pyarrow.parquet as pq

    CACHE_DIR = os.path.join("data", "hf_cache")
    PARQUET_URLS = [
        "https://huggingface.co/datasets/garystafford/deepfake-audio-detection"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "https://huggingface.co/datasets/garystafford/deepfake-audio-detection"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
    ]

    # Download if not cached
    os.makedirs(CACHE_DIR, exist_ok=True)
    for url in PARQUET_URLS:
        filename = url.split("/")[-1]
        local = os.path.join(CACHE_DIR, filename)
        if not os.path.isfile(local):
            import urllib.request
            log.info("Downloading %s ...", filename)
            urllib.request.urlretrieve(url, local)

    X_list, y_list = [], []
    skipped = 0
    for filename in ["0000.parquet", "0001.parquet"]:
        local = os.path.join(CACHE_DIR, filename)
        table = pq.read_table(local)
        df = table.to_pandas()
        log.info("Reading %s (%d rows)", filename, len(df))

        for _, row in df.iterrows():
            label = row["label"]
            if isinstance(label, (int, np.integer)):
                label = int(label)
            else:
                skipped += 1
                continue
            if label not in (0, 1):
                skipped += 1
                continue

            audio_col = row["audio"]
            try:
                if isinstance(audio_col, dict):
                    raw_bytes = audio_col.get("bytes")
                    if not raw_bytes:
                        skipped += 1
                        continue
                    waveform, src_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                else:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if waveform.size == 0:
                skipped += 1
                continue

            feat = extract_mel(waveform, src_sr)
            X_list.append(feat)
            y_list.append(label)

    log.info("HF dataset: %d samples (skipped %d)", len(X_list), skipped)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ── Conformer Architecture (same as before) ──────────────────────────────────

class ConvModule(layers.Layer):
    def __init__(self, d, ks=31, **kw):
        super().__init__(**kw)
        self.norm = layers.LayerNormalization()
        self.pw1 = layers.Conv1D(d*2, 1, padding="same")
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
        self.fc1 = layers.Dense(d*4, activation="swish")
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
        self.mhsa = layers.MultiHeadAttention(num_heads=h, key_dim=d//h, dropout=0.1)
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
    pe = tf.reshape(tf.stack([tf.sin(pos/scale), tf.cos(pos/scale)], axis=-1), (MAX_FRAMES, d))
    x = x + pe[tf.newaxis]
    x = layers.Dropout(0.1)(x)
    for i in range(blocks):
        x = ConfBlock(d, h, ks, name=f"conf_{i}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inp, outputs=out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Loading local Hebrew/Russian files (with augmentation) ===")
    X_local, y_local = load_local_files()

    log.info("=== Loading HuggingFace English dataset ===")
    X_hf, y_hf = load_hf_dataset()

    # Combine datasets
    X = np.concatenate([X_hf, X_local], axis=0)
    y = np.concatenate([y_hf, y_local], axis=0)
    log.info("Combined dataset: %d samples (HF=%d + local=%d)", len(X), len(X_hf), len(X_local))

    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    n_test = int(len(X) * 0.15)
    n_val = int(len(X) * 0.15)
    X_test, y_test = X[:n_test], y[:n_test]
    X_val, y_val = X[n_test:n_test+n_val], y[n_test:n_test+n_val]
    X_train, y_train = X[n_test+n_val:], y[n_test+n_val:]
    log.info("Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    # Class weights
    n_real = int(y_train.sum())
    n_fake = len(y_train) - n_real
    cw = {0: len(y_train)/(2*max(n_fake,1)), 1: len(y_train)/(2*max(n_real,1))}
    log.info("Classes: real=%d fake=%d weights=%s", n_real, n_fake, cw)

    # Build & compile
    model = build_model()
    model.summary(print_fn=log.info)

    lr = keras.optimizers.schedules.CosineDecayRestarts(1e-3, max(1, len(X_train)//16)*5, t_mul=2.0, m_mul=0.9)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True, verbose=1),
    ]

    # Train
    log.info("Training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=25, batch_size=16, class_weight=cw, callbacks=callbacks, verbose=1)

    # Evaluate
    results = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    acc = float((y_pred == y_test).mean())
    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY: {acc:.1%} ({int((y_pred==y_test).sum())}/{len(y_test)})")
    print(f"{'='*50}\n")

    # Save
    model.save("conformer_audio_model.keras")
    log.info("Model saved to conformer_audio_model.keras")


if __name__ == "__main__":
    main()
