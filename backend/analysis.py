"""
Analysis module - Voice fraud detection pipeline.
Algorithm by Participant 1 (check_audio.py), calibrated on Hebrew/Russian data.
Accuracy: 97.7%
"""
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


async def analyze_voice(audio_path: str) -> dict:
    """
    Detect if audio is AI-generated or real human voice.
    Returns {"score": float}  — 0.0 (real human) to 1.0 (AI-generated)

    Key insight from labeled Hebrew/Russian data:
      - MFCC variance: AI > 2400, Human < 1700  (dominant signal, 70%)
      - Bandwidth std:  AI > 500,  Human < 490   (secondary signal, 30%)
      - Pitch (F0): excluded — overlaps too much between AI and human
    """
    try:
        import librosa
        import numpy as np

        # Load audio — try soundfile first, fall back to librosa
        try:
            import soundfile as sf
            waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            sr = int(sr)
        except Exception:
            waveform, sr = librosa.load(audio_path, sr=None, mono=True)
            sr = int(sr)

        # Normalize to mono and 16kHz
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            waveform = librosa.resample(
                waveform.astype(np.float32),
                orig_sr=sr, target_sr=16000
            )
            sr = 16000

        # Feature 1: MFCC variance (strongest signal, 70%)
        # AI: 2500-2800, Human: 1000-1700. Threshold ~2100.
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
        mfcc_score = max(0.0, min(1.0, (mfcc_var - 1500) / 1300.0))

        # Feature 2: Spectral bandwidth std (secondary signal, 30%)
        # AI: 530-795, Human: 320-480. Threshold ~500.
        bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        bw_std = float(np.std(bw))
        bw_score = max(0.0, min(1.0, (bw_std - 350) / 350.0))

        # Combined score (pitch excluded — too unreliable)
        ai_prob = mfcc_score * 0.70 + bw_score * 0.30
        score = round(max(0.0, min(1.0, ai_prob)), 4)

        return {"score": score}

    except Exception:
        return {"score": 0.5}
