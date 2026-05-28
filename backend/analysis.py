"""
Analysis module - Voice and text fraud detection pipelines.
"""
import json
from pathlib import Path

import numpy as np

_PHRASES_PATH = Path(__file__).resolve().parent.parent / "data" / "phrases.json"


def _load_phrases() -> dict:
    """Load fraud phrases: {"he": [...], "ru": [...]}"""
    try:
        with open(_PHRASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if "he" in data or "ru" in data:
            return {
                lang: [p.lower() for p in phrases]
                for lang, phrases in data.items()
                if isinstance(phrases, list)
            }
        flat = [p.lower() for p in data.get("verification_phrases", [])]
        return {"en": flat}
    except FileNotFoundError:
        return {}


FRAUD_PHRASES = _load_phrases()


async def analyze_voice(audio_path: str) -> dict:
    """
    Analyze audio for signs of synthetic/cloned speech.
    Returns {"score": float}  — 0.0 (natural) to 1.0 (synthetic)
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=16000)

        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_std = float(np.std(zcr))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

        flatness_score = min(spectral_flatness * 10, 1.0)
        variance_penalty = max(0.0, 1.0 - mfcc_var / 50.0)
        zcr_score = max(0.0, 1.0 - zcr_std * 20)

        voice_score = round(
            flatness_score * 0.4 + variance_penalty * 0.35 + zcr_score * 0.25, 4
        )
        return {"score": min(max(voice_score, 0.0), 1.0)}

    except Exception:
        return {"score": 0.5}


async def analyze_text(audio_path: str) -> dict:
    """
    Transcribe audio and check for known fraud phrases.
    Returns {"score": float, "transcript": str, "language": str, "matched_phrases": list}
    """
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", compute_type="int8")
        segments, info = model.transcribe(audio_path, language=None)
        transcript = " ".join(seg.text for seg in segments).strip()
        language = getattr(info, "language", "unknown")
    except Exception:
        return {"score": 0.0, "transcript": "", "language": "unknown", "matched_phrases": []}

    transcript_lower = transcript.lower()
    matched: list = []

    phrases_to_check = FRAUD_PHRASES.get(language, [])
    if phrases_to_check:
        for phrase in phrases_to_check:
            if phrase in transcript_lower:
                matched.append(phrase)
    else:
        for phrases in FRAUD_PHRASES.values():
            for phrase in phrases:
                if phrase in transcript_lower and phrase not in matched:
                    matched.append(phrase)

    if len(matched) == 0:
        score = 0.0
    elif len(matched) == 1:
        score = 0.6
    else:
        score = min(0.85 + (len(matched) - 2) * 0.05, 1.0)

    return {
        "score": score,
        "transcript": transcript,
        "language": language,
        "matched_phrases": matched,
    }
