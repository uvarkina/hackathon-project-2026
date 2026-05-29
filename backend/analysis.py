"""
Analysis module — fast AI voice heuristic.

Использует формулу из check_audio.py:
    AI_probability = 0.70 * MFCC_variance_score + 0.30 * bandwidth_std_score
На размеченных ивритских/русских данных давала 97.7% точности.

Никаких TensorFlow / PyTorch / sklearn — только librosa.
Время на 3-секундный чанк: ~50-100 мс.
"""
import os
import warnings
import asyncio

warnings.filterwarnings("ignore")


def _score_heuristic(wav, sr) -> float:
    """check_audio.py: 70% MFCC variance + 30% bandwidth std."""
    import numpy as np
    import librosa

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000

    # MFCC variance: AI > 2400, Human < 1700. Threshold ~2100.
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
    mfcc_score = max(0.0, min(1.0, (mfcc_var - 1500) / 1300.0))

    # Spectral bandwidth std: AI > 500, Human < 490. Threshold ~500.
    bw = librosa.feature.spectral_bandwidth(y=wav, sr=sr)[0]
    bw_std = float(np.std(bw))
    bw_score = max(0.0, min(1.0, (bw_std - 350) / 350.0))

    return round(mfcc_score * 0.70 + bw_score * 0.30, 4)


async def analyze_voice(audio_path: str) -> dict:
    """
    Detect if audio is AI-generated or real human voice.
    Returns {"score": float} — 0.0 (real human) to 1.0 (AI-generated).
    """
    try:
        try:
            import soundfile as sf
            wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            sr = int(sr)
        except Exception:
            import librosa
            wav, sr = librosa.load(audio_path, sr=None, mono=True)
            sr = int(sr)

        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(None, _score_heuristic, wav, sr)
        return {"score": max(0.0, min(1.0, score))}

    except Exception:
        return {"score": 0.5}
