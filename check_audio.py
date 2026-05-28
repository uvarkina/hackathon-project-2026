"""Quick AI voice detection script."""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

def analyze(waveform, sr):
    import librosa, numpy as np
    if waveform.ndim > 1: waveform = waveform.mean(axis=1)
    if sr != 16000:
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        sr = 16000

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)))
    f0, _, _ = librosa.pyin(waveform, fmin=60, fmax=400, sr=sr, frame_length=2048)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0_valid) > 5:
        f0_std = float(np.std(f0_valid))
        f0_range = float(np.ptp(f0_valid))
        pitch_score = max(0.0, min(1.0, 1.0 - (f0_std - 20) / 50.0))
        range_score = max(0.0, min(1.0, 1.0 - (f0_range - 100) / 250.0))
    else:
        f0_std, f0_range, pitch_score, range_score = 0, 0, 0.5, 0.5

    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
    mfcc_score = max(0.0, min(1.0, (mfcc_var - 1500) / 1500.0))

    bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    bw_std = float(np.std(bw))
    bw_score = max(0.0, min(1.0, (bw_std - 350) / 400.0))

    ai_prob = pitch_score * 0.40 + range_score * 0.25 + mfcc_score * 0.25 + bw_score * 0.10
    return max(0.0, min(1.0, ai_prob)) * 100, f0_std, f0_range, mfcc_var, bw_std

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_audio.py <file_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if os.path.splitext(f)[1].lower() in ('.mp3','.wav','.flac','.m4a')]
    else:
        files = [target]

    import soundfile as sf, librosa
    print(f"\n{'File':<45} {'AI%':>5}  {'Verdict':<20} {'F0std':>6} {'F0rng':>6} {'MFCC':>7} {'BW':>5}")
    print("-" * 105)

    for path in files:
        name = os.path.basename(path)
        try:
            try:
                w, sr = sf.read(path, dtype="float32", always_2d=False)
            except Exception:
                w, sr = librosa.load(path, sr=None, mono=True, res_type="soxr_hq")
                sr = int(sr)
            ai_pct, f0s, f0r, mfcc, bw = analyze(w, sr)
            verdict = "AI-GENERATED" if ai_pct >= 60 else ("SUSPICIOUS" if ai_pct >= 40 else "REAL human")
            print(f"{name:<45} {ai_pct:>5.1f}  {verdict:<20} {f0s:>6.1f} {f0r:>6.1f} {mfcc:>7.0f} {bw:>5.0f}")
        except Exception as e:
            print(f"{name:<45} ERROR: {e}")

    print()

if __name__ == "__main__":
    main()
