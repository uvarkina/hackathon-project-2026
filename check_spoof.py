"""
check_spoof.py — sanity-check a PRETRAINED HuggingFace anti-spoofing model
on your own audio files (no training required).

Model: MelodyMachine/Deepfake-audio-detection-V2  (wav2vec2-base, labels 0=fake/1=real)
       override with env SPOOF_MODEL=<repo_id>

One-time setup (downloads ~360 MB the first run):
    pip install torch transformers soundfile librosa --break-system-packages

Usage:
    python check_spoof.py text_fraud_recognition
    python check_spoof.py text_fraud_recognition/hebrew_3_ai.mp3

AI% = probability of the "fake" class. >=60 AI, 40-60 suspicious, <40 human.
"""
import os, sys, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np

MODEL_ID = os.environ.get("SPOOF_MODEL", "MelodyMachine/Deepfake-audio-detection-V2")
TARGET_SR = 16000


def load_audio(path):
    try:
        import soundfile as sf
        w, sr = sf.read(path, dtype="float32", always_2d=False)
        sr = int(sr)
    except Exception:
        import librosa
        w, sr = librosa.load(path, sr=None, mono=True)
        sr = int(sr)
    if getattr(w, "ndim", 1) > 1:
        w = w.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return w.astype(np.float32), TARGET_SR


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_spoof.py <file_or_folder>")
        sys.exit(1)

    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"Loading {MODEL_ID} (first run downloads weights)...")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    fake_idx = next((i for i, l in id2label.items()
                     if l in ("fake", "spoof", "ai", "synthetic")), 0)
    print(f"labels={id2label}  fake_idx={fake_idx}\n")

    target = sys.argv[1]
    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if os.path.splitext(f)[1].lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm")]
    else:
        files = [target]

    print(f"{'file':<40}{'AI%':>6}  {'verdict':<12}{'AI%(first3s)':>13}")
    print("-" * 76)
    for path in files:
        name = os.path.basename(path)
        try:
            w, sr = load_audio(path)

            def ai_pct(wav):
                inp = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
                with torch.no_grad():
                    p = torch.softmax(model(**inp).logits, dim=-1)[0]
                return float(p[fake_idx].item()) * 100

            full = ai_pct(w)
            first3 = ai_pct(w[:sr * 3]) if w.size > sr * 1 else full
            verdict = "AI" if full >= 60 else ("SUSPICIOUS" if full >= 40 else "human")
            print(f"{name:<40}{full:>6.1f}  {verdict:<12}{first3:>13.1f}")
        except Exception as e:
            print(f"{name:<40} ERROR: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    main()
