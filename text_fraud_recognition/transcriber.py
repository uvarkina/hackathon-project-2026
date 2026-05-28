import wave
import numpy as np
from faster_whisper import WhisperModel
from dotenv import load_dotenv
load_dotenv()

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model


def _read_wav(file_path: str):
    """Read a WAV file without ffmpeg using Python stdlib wave module.
    Returns (float32 numpy array at original sample rate, sample_rate)."""
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.float32)

    # take first channel if stereo
    if n_channels > 1:
        audio = audio[::n_channels]

    return audio, frame_rate


def _run_forced(model, audio_input, lang, extra_kwargs):
    """Transcribe with a forced language. Returns (text, segments_list)."""
    segs_gen, _ = model.transcribe(audio_input, language=lang, **extra_kwargs)
    parts = []
    segs_list = []
    for seg in segs_gen:
        t = seg.text.strip()
        parts.append(t)
        segs_list.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": t,
        })
    return " ".join(parts), segs_list


def _has_hebrew(text: str) -> bool:
    return any('֐' <= c <= '׿' for c in text)


def _has_cyrillic(text: str) -> bool:
    return any('Ѐ' <= c <= 'ӿ' for c in text)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a short audio clip (~3 seconds) using faster-whisper.
    Reads WAV directly via numpy — no ffmpeg required.
    Supports Hebrew ("he") and Russian ("ru") only.
    Never outputs Arabic or other languages.
    """
    model = _get_model()

    # Try to load WAV directly; fall back to file path (for mp3/webm)
    try:
        audio_input, sr = _read_wav(audio_path)
        base_kwargs = dict(
            sampling_rate=sr,
            beam_size=1,
            vad_filter=True,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
    except Exception:
        audio_input = audio_path
        base_kwargs = dict(
            beam_size=1,
            vad_filter=True,
        )

    # ── Step 1: detect language without caring about transcription output ──
    # We run auto-detect first just to know which of he/ru to force.
    try:
        _, info_detect = model.transcribe(
            audio_input,
            language=None,
            **base_kwargs,
        )
        auto_lang = info_detect.language  # e.g. "he", "ru", "ar", "en" …
    except Exception:
        auto_lang = "he"

    # Map auto-detected language to the one we support
    if auto_lang == "ru":
        force_lang = "ru"
    else:
        # Hebrew, Arabic, Farsi, and other Semitic/Middle-Eastern languages
        # all map to Hebrew — it's the primary language of our app
        force_lang = "he"

    # ── Step 2: transcribe with the forced language ──
    text, segments_list = _run_forced(model, audio_input, force_lang, base_kwargs)

    # ── Step 3: script validation — if text doesn't match the script, discard ──
    if force_lang == "he" and not _has_hebrew(text):
        # Maybe it's actually Russian — try once more
        text_ru, segs_ru = _run_forced(model, audio_input, "ru", base_kwargs)
        if _has_cyrillic(text_ru):
            return {"text": text_ru, "language": "ru", "segments": segs_ru}
        # Nothing recognisable — return empty (silence / noise)
        return {"text": "", "language": "he", "segments": []}

    if force_lang == "ru" and not _has_cyrillic(text):
        # Maybe it's Hebrew — try once more
        text_he, segs_he = _run_forced(model, audio_input, "he", base_kwargs)
        if _has_hebrew(text_he):
            return {"text": text_he, "language": "he", "segments": segs_he}
        return {"text": "", "language": "ru", "segments": []}

    return {
        "text": text,
        "language": force_lang,
        "segments": segments_list,
    }


if __name__ == "__main__":
    result = transcribe_audio("hebrew_3_ai.mp3")
    print(f"Language: {result['language']}")
    print(f"Text: {result['text']}")
