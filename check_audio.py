"""
check_audio.py — AI voice detection calibrated on Hebrew/Russian speech data.

Usage:
    python check_audio.py <file_or_folder>
    python check_audio.py text_fraud_recognition/hebrew_3_ai.mp3
    python check_audio.py text_fraud_recognition
"""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def analyze(waveform, sr):
    """
    Detect AI-generated speech using features calibrated on real data.

    Key insight from our labeled data:
      - MFCC variance: AI > 2400, Human < 1700 (strongest signal, no overlap)
      - Bandwidth std: AI > 500, Human < 490 (good secondary signal)
      - Pitch (F0_std): overlaps too much between AI and human — low weight

    Returns: (ai_probability_pct, f0_std, f0_range, mfcc_var, bw_std)
    """
    import librosa
    import numpy as np

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != 16000:
        waveform = librosa.resample(waveform.astype(np.float32),
                                     orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        sr = 16000

    # ── 1. MFCC Variance (strongest signal) ───────────────────────────────
    # AI: 2500-2800, Human: 1000-1700. Threshold at ~2100.
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
    # Score: 0 at mfcc_var=1500, 1.0 at mfcc_var=2800
    mfcc_score = max(0.0, min(1.0, (mfcc_var - 1500) / 1300.0))

    # ── 2. Spectral Bandwidth Std (secondary signal) ──────────────────────
    # AI: 530-795, Human: 320-480. Threshold at ~500.
    bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    bw_std = float(np.std(bw))
    # Score: 0 at bw=350, 1.0 at bw=700
    bw_score = max(0.0, min(1.0, (bw_std - 350) / 350.0))

    # ── 3. Pitch variance (weak signal — overlaps) ────────────────────────
    f0, _, _ = librosa.pyin(waveform, fmin=60, fmax=400, sr=sr, frame_length=2048)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0_valid) > 5:
        f0_std = float(np.std(f0_valid))
        f0_range = float(np.ptp(f0_valid))
    else:
        f0_std, f0_range = 30.0, 150.0  # neutral defaults

    # ── Combined Score ────────────────────────────────────────────────────
    # MFCC is dominant (70%), bandwidth secondary (30%)
    # Pitch excluded — too unreliable with this data
    ai_probability = mfcc_score * 0.70 + bw_score * 0.30
    ai_probability = max(0.0, min(1.0, ai_probability))

    return ai_probability * 100, f0_std, f0_range, mfcc_var, bw_std


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_audio.py <file_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if os.path.splitext(f)[1].lower() in ('.mp3', '.wav', '.flac', '.m4a')]
    else:
        files = [target]

    import soundfile as sf, librosa

    print(f"\n{'File':<45} {'AI%':>5}  {'Verdict':<20} {'MFCC':>7} {'BW':>5} {'F0std':>6}")
    print("-" * 100)

    for path in files:
        name = os.path.basename(path)
        try:
            try:
                w, sr = sf.read(path, dtype="float32", always_2d=False)
            except Exception:
                w, sr = librosa.load(path, sr=None, mono=True, res_type="soxr_hq")
                sr = int(sr)
            ai_pct, f0s, f0r, mfcc, bw = analyze(w, sr)
            if ai_pct >= 60:
                verdict = "AI-GENERATED"
            elif ai_pct >= 40:
                verdict = "SUSPICIOUS"
            else:
                verdict = "REAL human"
            print(f"{name:<45} {ai_pct:>5.1f}  {verdict:<20} {mfcc:>7.0f} {bw:>5.0f} {f0s:>6.1f}")
        except Exception as e:
            print(f"{name:<45} ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
