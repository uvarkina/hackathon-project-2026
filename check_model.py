"""
check_model.py — run YOUR retrained conformer (conformer_audio_model.keras)
on audio files and print AI%.

Convention after retraining: model output = P(AI) directly (0=human, 1=AI).

Run inside the venv:
    python check_model.py text_fraud_recognition
    python check_model.py some_file.mp3
"""
import os, sys, zipfile, tempfile, warnings, glob
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TARGET_SR = 16000
N_MELS = 80
MAX_FRAMES = 400
MODEL_PATH = "conformer_audio_model.keras"


class ConvModule(layers.Layer):
    def __init__(self, d, ks=31, **kw):
        super().__init__(**kw)
        self.norm = layers.LayerNormalization(); self.pw1 = layers.Conv1D(d*2, 1, padding="same")
        self.dw = layers.DepthwiseConv1D(ks, padding="same"); self.bn = layers.BatchNormalization()
        self.pw2 = layers.Conv1D(d, 1, padding="same"); self.drop = layers.Dropout(0.1)
    def call(self, x, training=False):
        r = x; x = self.norm(x); x = self.pw1(x); x, g = tf.split(x, 2, axis=-1); x = x*tf.sigmoid(g)
        x = self.dw(x); x = self.bn(x, training=training); x = tf.nn.silu(x); x = self.pw2(x)
        return r + self.drop(x, training=training)
class FFModule(layers.Layer):
    def __init__(self, d, **kw):
        super().__init__(**kw); self.norm = layers.LayerNormalization()
        self.fc1 = layers.Dense(d*4, activation="swish"); self.drop = layers.Dropout(0.1); self.fc2 = layers.Dense(d)
    def call(self, x, training=False):
        r = x; x = self.norm(x); x = self.fc1(x); x = self.drop(x, training=training); x = self.fc2(x); return r + 0.5*x
class ConfBlock(layers.Layer):
    def __init__(self, d, h, ks=31, **kw):
        super().__init__(**kw); self.ff1 = FFModule(d); self.norm = layers.LayerNormalization()
        self.mhsa = layers.MultiHeadAttention(num_heads=h, key_dim=d//h, dropout=0.1)
        self.drop = layers.Dropout(0.1); self.conv = ConvModule(d, ks); self.ff2 = FFModule(d); self.ln = layers.LayerNormalization()
    def call(self, x, training=False):
        x = self.ff1(x, training=training); r = x; xn = self.norm(x)
        x = r + self.drop(self.mhsa(xn, xn, training=training), training=training)
        x = self.conv(x, training=training); x = self.ff2(x, training=training); return self.ln(x)

def build_model(d=144, h=4, blocks=4, ks=31):
    inp = keras.Input(shape=(MAX_FRAMES, N_MELS))
    x = layers.Dense(d)(inp)
    pos = tf.cast(tf.range(MAX_FRAMES), tf.float32)[:, tf.newaxis]
    dims = tf.cast(tf.range(0, d, 2), tf.float32)[tf.newaxis, :]
    scale = tf.pow(10000.0, dims / tf.cast(d, tf.float32))
    pe = tf.reshape(tf.stack([tf.sin(pos/scale), tf.cos(pos/scale)], axis=-1), (MAX_FRAMES, d))
    x = x + pe[tf.newaxis]; x = layers.Dropout(0.1)(x)
    for i in range(blocks): x = ConfBlock(d, h, ks, name=f"conf_{i}")(x)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(0.2)(x); out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)

def load_audio(path):
    try:
        import soundfile as sf
        w, sr = sf.read(path, dtype="float32", always_2d=False); sr = int(sr)
    except Exception:
        import librosa
        w, sr = librosa.load(path, sr=None, mono=True); sr = int(sr)
    if getattr(w, "ndim", 1) > 1: w = w.mean(axis=1)
    return w.astype(np.float32), sr

def extract_mel(w, sr):
    import librosa
    if sr != TARGET_SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)
    mel = librosa.feature.melspectrogram(y=w, sr=TARGET_SR, n_fft=512, hop_length=160,
                                         win_length=400, n_mels=N_MELS, fmin=0, fmax=8000, power=2.0)
    lm = librosa.power_to_db(mel, ref=np.max, top_db=80.0); std = lm.std()
    lm = (lm - lm.mean()) / (std if std > 1e-8 else 1.0); feat = lm.T.astype(np.float32)
    if feat.shape[0] >= MAX_FRAMES: feat = feat[:MAX_FRAMES]
    else: feat = np.concatenate([feat, np.zeros((MAX_FRAMES - feat.shape[0], N_MELS), np.float32)])
    return feat[None, ...]

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_model.py <file_or_folder>"); sys.exit(1)
    m = build_model()
    with zipfile.ZipFile(MODEL_PATH) as z:
        wb = z.read("model.weights.h5")
    wp = os.path.join(tempfile.gettempdir(), "cm.weights.h5"); open(wp, "wb").write(wb)
    m.load_weights(wp)
    print("Model loaded.\n")

    target = sys.argv[1]
    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if os.path.splitext(f)[1].lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg")]
    else:
        files = [target]

    print(f"{'file':<40}{'AI%':>6}  verdict   (NOTE: local files were in training)")
    print("-" * 78)
    for path in files:
        name = os.path.basename(path)
        try:
            w, sr = load_audio(path)
            ai = float(m.predict(extract_mel(w, sr), verbose=0).flatten()[0]) * 100
            verdict = "AI" if ai >= 60 else ("susp" if ai >= 40 else "human")
            print(f"{name:<40}{ai:>6.1f}  {verdict}")
        except Exception as e:
            print(f"{name:<40} ERROR: {type(e).__name__}: {e}")
    print()

if __name__ == "__main__":
    main()
