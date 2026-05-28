from faster_whisper import WhisperModel
from dotenv import load_dotenv
load_dotenv()


# Load model once at module level for efficiency across multiple calls
_model = None


def _get_model():
    """Lazy-load the Whisper model (tiny) on first use."""
    global _model
    if _model is None:
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a short audio clip (~3 seconds) using faster-whisper.

    Auto-detects language between Hebrew ("he") and Russian ("ru").

    Args:
        audio_path: Path to the audio file (wav, mp3, etc.)

    Returns:
        dict with keys:
            - text: Full transcribed text
            - language: Detected language code ("he" or "ru")
            - segments: List of segment dicts with start, end, and text
    """
    model = _get_model()

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=1,              # faster for short clips
        language=None,            # auto-detect
        vad_filter=False,         # skip VAD for 3-second clips
    )

    detected_language = info.language  # e.g. "he", "ru", "en"

    # Constrain to expected languages
    if detected_language not in ("he", "ru"):
        # Default to the higher-probability one from detection
        he_prob = info.language_probability if detected_language == "he" else 0.0
        ru_prob = info.language_probability if detected_language == "ru" else 0.0
        detected_language = "he" if he_prob >= ru_prob else detected_language

    segments_list = []
    full_text_parts = []

    for segment in segments_gen:
        segments_list.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        })
        full_text_parts.append(segment.text.strip())

    full_text = " ".join(full_text_parts)

    return {
        "text": full_text,
        "language": detected_language,
        "segments": segments_list,
    }


if __name__ == "__main__":
    import sys


    result = transcribe_audio('hebrew_3_ai.mp3')
    print(f"Language: {result['language']}")
    print(f"Text: {result['text']}")
    print(f"Segments: {result['segments']}")
