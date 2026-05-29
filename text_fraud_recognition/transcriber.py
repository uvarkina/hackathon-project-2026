import wave
import numpy as np
from faster_whisper import WhisperModel
from dotenv import load_dotenv
load_dotenv()

_model = None

# initial_prompt helps Whisper stay in Hebrew mode and improves recognition
_HE_PROMPT = "שיחה בעברית. זיהוי הונאה."  # "Conversation in Hebrew. Fraud detection."


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
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
    Always transcribes in Hebrew ("he") only.
    """
    model = _get_model()

    # Try to load WAV directly; fall back to file path (for mp3/webm)
    try:
        audio_input, sr = _read_wav(audio_path)
        base_kwargs = dict(
            sampling_rate=sr,
            beam_size=5,
            vad_filter=False,          # don't filter — Hebrew phonemes can be cut by VAD
            condition_on_previous_text=False,
            initial_prompt=_HE_PROMPT, # hint Whisper it's Hebrew
        )
    except Exception:
        audio_input = audio_path
        base_kwargs = dict(
            beam_size=5,
            vad_filter=False,
            initial_prompt=_HE_PROMPT,
        )

    # Always force Hebrew — no auto-detection, no Arabic, no Russian
    text, segments_list = _run_forced(model, audio_input, "he", base_kwargs)

    # If output contains no Hebrew characters — it's silence or noise, discard
    if not _has_hebrew(text):
        text = ""
        segments_list = []

    return {
        "text": text,
        "language": "he",
        "segments": segments_list,
    }


if __name__ == "__main__":
    result = transcribe_audio("hebrew_3_ai.mp3")
    print(f"Language: {result['language']}")
    print(f"Text: {result['text']}")
