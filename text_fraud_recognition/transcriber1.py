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
        # small >> base for Hebrew; int8 keeps it fast on CPU
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def _decode_audio_av(file_path: str):
    """Decode any audio format (webm/opus/mp3/…) to 16 kHz mono float32 using PyAV.
    PyAV bundles its own libav — no system ffmpeg required."""
    import av
    container = av.open(file_path)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(audio=0):
        frame.pts = None  # required by the resampler
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray()[0])
    for out in resampler.resample(None):  # flush
        chunks.append(out.to_ndarray()[0])
    container.close()
    audio = np.concatenate(chunks) if chunks else np.zeros(160, dtype=np.float32)
    return audio, 16000


def _read_wav(file_path: str):
    """Read a PCM WAV file via stdlib (no external deps)."""
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

    if n_channels > 1:
        audio = audio[::n_channels]

    return audio, frame_rate


def _load_audio(file_path: str):
    """Try WAV stdlib first (zero deps), then PyAV for webm/mp3/etc."""
    try:
        return _read_wav(file_path)
    except Exception:
        pass
    return _decode_audio_av(file_path)


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
    return any('א' <= c <= 'ת' for c in text)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a short audio clip (~3 seconds) using faster-whisper.
    Decodes any format (including webm from browser) via PyAV — no system ffmpeg needed.
    Always forces Hebrew output.
    """
    model = _get_model()

    audio_input, sr = _load_audio(audio_path)
    base_kwargs = dict(
        sampling_rate=sr,
        beam_size=5,
        temperature=0.0,               # deterministic; reduces random Russian output
        vad_filter=False,              # VAD can cut short Hebrew phonemes
        condition_on_previous_text=False,
        initial_prompt=_HE_PROMPT,
    )

    # Always force Hebrew — no auto-detection
    text, segments_list = _run_forced(model, audio_input, "he", base_kwargs)

    # Discard if no Hebrew script (silence, noise, or hallucination)
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
