"""
Analysis module - Voice and text fraud detection pipelines.
"""
import json
from pathlib import Path

import numpy as np

# Fraud indicator phrases loaded from data/phrases.json
_PHRASES_PATH = Path(__file__).resolve().parent.parent / "data" / "phrases.json"


def _load_phrases() -> list[str]:
    """Load known fraud phrases from the data directory."""
    try:
        with open(_PHRASES_PATH) as f:
            data = json.load(f)
        return [p.lower() for p in data.get("verification_phrases", [])]
    except FileNotFoundError:
        return []


FRAUD_PHRASES = _load_phrases()


async def analyze_voice(audio_path: str) -> dict:
    """
    Analyze audio features for signs of synthetic/cloned speech.

    Returns:
        {"score": float}  — 0.0 (natural) to 1.0 (synthetic)
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=16000)

        # Feature extraction for voice authenticity
        # Spectral flatness: synthetic voices tend to have higher flatness
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        # Zero crossing rate: synthetic speech often has more uniform ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_std = float(np.std(zcr))

        # MFCC variance: natural speech has more variation
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

        # Heuristic scoring (placeholder for trained model)
        # Higher spectral flatness + lower MFCC variance = more likely synthetic
        flatness_score = min(spectral_flatness * 10, 1.0)
        variance_penalty = max(0, 1.0 - mfcc_var / 50.0)
        zcr_score = max(0, 1.0 - zcr_std * 20)

        voice_score = round(
            flatness_score * 0.4 + variance_penalty * 0.35 + zcr_score * 0.25, 4
        )
        voice_score = min(max(voice_score, 0.0), 1.0)

        return {"score": voice_score}

    except Exception:
        # If analysis fails, return neutral score
        return {"score": 0.5}


async def analyze_text(audio_path: str) -> dict:
    """
    Transcribe audio and check for known fraud/social-engineering phrases.

    Returns:
        {"score": float, "transcript": str, "matched_phrases": list[str]}
    """
    transcript = ""
    matched_phrases: list[str] = []

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base", compute_type="int8")
        segments, _ = model.transcribe(audio_path, language="en")
        transcript = " ".join(seg.text for seg in segments).strip()
    except Exception:
        # If transcription fails, return neutral
        return {"score": 0.0, "transcript": "", "matched_phrases": []}

    # Check transcript against known fraud phrases
    transcript_lower = transcript.lower()
    for phrase in FRAUD_PHRASES:
        if phrase in transcript_lower:
            matched_phrases.append(phrase)

    # Score based on how many fraud phrases matched
    if not FRAUD_PHRASES:
        text_score = 0.0
    else:
        text_score = round(len(matched_phrases) / max(len(FRAUD_PHRASES), 1), 4)
    text_score = min(text_score, 1.0)

    return {
        "score": text_score,
        "transcript": transcript,
        "matched_phrases": matched_phrases,
    }
