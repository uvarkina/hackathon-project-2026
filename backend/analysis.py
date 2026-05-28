"""
Analysis module - Voice fraud detection pipeline.
Uses calibrated algorithm from check_audio.py (Participant 1).
"""
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


async def analyze_voice(audio_path: str) -> dict:
    """
    Detect if audio is AI-generated or real human voice.
    Returns {"score": float}  — 0.0 (real human) to 1.0 (AI-generated)

    Algorithm by Participant 1 (check_audio.py):
    - Pitch stability (f0 std, range) — AI voices have unnatural pitch
    - MFCC variance — AI voices have less variation
    - Spectral bandwidth — AI voices have narrower bandwidth
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

        # Feature 1: Pitch analysis — AI voices have unnaturally stable pitch
        f0, _, _ = librosa.pyin(waveform, fmin=60, fmax=400, sr=sr, frame_length=2048)
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_valid) > 5:
            f0_std = float(np.std(f0_valid))
            f0_range = float(np.ptp(f0_valid))
            pitch_score = max(0.0, min(1.0, 1.0 - (f0_std - 20) / 50.0))
            range_score = max(0.0, min(1.0, 1.0 - (f0_range - 100) / 250.0))
        else:
            pitch_score, range_score = 0.5, 0.5

        # Feature 2: MFCC variance — AI voices have less variation
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
        mfcc_score = max(0.0, min(1.0, (mfcc_var - 1500) / 1500.0))

        # Feature 3: Spectral bandwidth — AI voices have narrower spectrum
        bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        bw_std = float(np.std(bw))
        bw_score = max(0.0, min(1.0, (bw_std - 350) / 400.0))

        # Weighted combination (calibrated by Participant 1)
        ai_prob = (
            pitch_score * 0.40 +
            range_score * 0.25 +
            mfcc_score  * 0.25 +
            bw_score    * 0.10
        )
        score = round(max(0.0, min(1.0, ai_prob)), 4)
        return {"score": score}

    except Exception:
        return {"score": 0.5}
