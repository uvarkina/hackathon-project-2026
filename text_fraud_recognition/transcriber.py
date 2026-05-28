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


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a short audio clip (~3 seconds) using faster-whisper.
    Reads WAV directly via numpy — no ffmpeg required.
    Auto-detects language between Hebrew ("he") and Russian ("ru").
    """
    model = _get_model()

    try:
        audio_array, sr = _read_wav(audio_path)
        segments_gen, info = model.transcribe(
            audio_array,
            sampling_rate=sr,
            beam_size=1,
            language=None,          # авто-определение
            vad_filter=True,        # пропускать тишину
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
    except Exception:
        segments_gen, info = model.transcribe(
            audio_path,
            beam_size=1,
            language=None,
            vad_filter=True,
        )

    detected_language = info.language if info.language in ("he", "ru") else "ru"

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

    # Filter hallucinations: if forced language is ru/he but text has no matching chars — discard
    if detected_language == "ru":
        has_script = any('Ѐ' <= c <= 'ӿ' for c in full_text)
    elif detected_language == "he":
        has_script = any('֐' <= c <= '׿' for c in full_text)
    else:
        has_script = True

    if not has_script:
        full_text = ""
        segments_list = []

    return {
        "text": full_text,
        "language": detected_language,
        "segments": segments_list,
    }


if __name__ == "__main__":
    result = transcribe_audio("hebrew_3_ai.mp3")
    print(f"Language: {result['language']}")
    print(f"Text: {result['text']}")
