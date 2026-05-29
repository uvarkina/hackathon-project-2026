import wave
import numpy as np
from faster_whisper import WhisperModel
from dotenv import load_dotenv
load_dotenv()

_model = None
_suppress_cyrillic = None  # cached list of cyrillic token IDs

# initial_prompt отключён: Whisper эхает его в выходе ("זיהוי הונאה" повторяется
# каждые 3 секунды). Язык всё равно жёстко задан через language="he".
_HE_PROMPT = None
_PROMPT_FRAGMENTS = ("זיהוי הונאה", "שיחה בעברית")  # на случай если когда-то вернём


def _get_cyrillic_token_ids(model):
    """
    Whisper мультилингвальный и на нечётких ивритских фонемах "соскальзывает"
    в русский. Подавляем все токены, содержащие кириллицу — это убирает
    галлюцинации вида "давайте", "просто" и т.п. посреди иврита.
    Кэшируется после первого вызова.
    """
    global _suppress_cyrillic
    if _suppress_cyrillic is not None:
        return _suppress_cyrillic
    bad = []
    try:
        tok = model.hf_tokenizer
        vocab_size = tok.get_vocab_size()
        for tid in range(vocab_size):
            s = tok.decode([tid]) or ""
            if any('Ѐ' <= c <= 'ӿ' for c in s):
                bad.append(tid)
    except Exception as e:
        print(f"[transcriber] cyrillic-suppress build failed: {e}")
    print(f"[transcriber] suppressing {len(bad)} Cyrillic tokens")
    _suppress_cyrillic = bad
    return bad


# ── Fix 2: small >> base for Hebrew recognition ──────────────────────────────
def _get_model():
    global _model
    if _model is None:
        # "small" has much better Hebrew phoneme coverage than "base";
        # int8 keeps CPU latency acceptable (~2-4 s per 3-s chunk)
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


# ── Fix 1: decode any format via PyAV (no system ffmpeg needed) ──────────────
def _decode_audio_av(file_path: str):
    """Decode webm/opus/mp3/m4a/… to 16 kHz mono float32 using PyAV.
    PyAV bundles its own libav — no system ffmpeg required."""
    import av
    container = av.open(file_path)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(audio=0):
        frame.pts = None          # required by the resampler
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray()[0])
    for out in resampler.resample(None):  # flush
        chunks.append(out.to_ndarray()[0])
    container.close()
    audio = np.concatenate(chunks) if chunks else np.zeros(160, dtype=np.float32)
    return audio, 16000


def _read_wav(file_path: str):
    """Read a PCM WAV file via stdlib (zero external deps)."""
    with wave.open(file_path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate  = wf.getframerate()
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


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz via linear interpolation (no external deps)."""
    if sr == 16000:
        return audio
    target_len = int(len(audio) * 16000 / sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _load_audio(file_path: str):
    """Try WAV stdlib first; fall back to PyAV for webm/opus/mp3/m4a/…"""
    try:
        audio, sr = _read_wav(file_path)
        return _resample_to_16k(audio, sr), 16000
    except Exception:
        pass
    return _decode_audio_av(file_path)   # Fix 1: PyAV handles everything else


def _run_forced(model, audio_input, lang, extra_kwargs):
    """Transcribe with a forced language. Returns (text, segments_list)."""
    # NOTE: faster_whisper.transcribe() does NOT accept 'sampling_rate' —
    # the numpy array must already be at 16 kHz before this call.
    segs_gen, _ = model.transcribe(audio_input, language=lang, **extra_kwargs)
    parts    = []
    segs_list = []
    for seg in segs_gen:
        t = seg.text.strip()
        parts.append(t)
        segs_list.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end,   3),
            "text":  t,
        })
    return " ".join(parts), segs_list


# ── Fix 3: check only actual Hebrew letters (א–ת), not the broad block ───────
def _has_hebrew(text: str) -> bool:
    return any('א' <= c <= 'ת' for c in text)


def _has_cyrillic(text: str) -> bool:
    return any('Ѐ' <= c <= 'ӿ' for c in text)


def _looks_like_hallucination(text: str) -> bool:
    """
    Heuristic: Whisper often glitches on silence/noise producing one repeated
    word/phrase, e.g. "עוד עוד עוד" or echoing the initial_prompt verbatim.
    """
    if not text:
        return False
    # 1. echo of the Hebrew prompt
    for frag in _PROMPT_FRAGMENTS:
        if text.count(frag) >= 2:
            return True
    # 2. one word filling the whole output
    words = [w for w in text.split() if w.strip()]
    if len(words) >= 4:
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) >= 0.6:
            return True
    return False


def _strip_prompt_echo(text: str) -> str:
    """If the prompt slipped through once or twice, scrub it out."""
    for frag in _PROMPT_FRAGMENTS:
        text = text.replace(frag, "")
    # collapse extra spaces and stray punctuation left behind
    return " ".join(text.split()).strip(" .,")


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a short audio clip (~3 seconds) using faster-whisper.
    • Decodes any format (webm/opus/mp3/wav) via PyAV — no system ffmpeg needed.
    • Always forces Hebrew output.
    • Filters silence-induced hallucinations.
    """
    model = _get_model()
    audio_input, _ = _load_audio(audio_path)

    # -1 is faster-whisper's default (non-speech tokens). We add Cyrillic on top.
    suppress = [-1] + _get_cyrillic_token_ids(model)

    base_kwargs = dict(
        beam_size=3,                          # 5 → 3: faster, near-identical quality
        best_of=1,                            # default 5 → 1: skips extra decoding passes
        temperature=0.0,                      # deterministic → fewer Cyrillic hallucinations
        suppress_tokens=suppress,             # hard block any Russian leakage
        # Anti-hallucination on silence/noise:
        no_speech_threshold=0.6,              # Whisper will skip "tishina" segments
        log_prob_threshold=-1.0,              # drop low-confidence segments
        compression_ratio_threshold=2.4,      # drop pathologically repetitive output
        vad_filter=True,                      # VAD with soft params (was disabled)
        vad_parameters=dict(
            min_silence_duration_ms=300,      # short enough not to chop Hebrew phonemes
            speech_pad_ms=200,
        ),
        condition_on_previous_text=False,
        # initial_prompt намеренно НЕ передаётся — Whisper эхает его в выходе.
    )

    # Always force Hebrew — no auto-detection, no Arabic, no Russian
    text, segments_list = _run_forced(model, audio_input, "he", base_kwargs)

    # Strip leftover prompt echo from the very rare case it slips through.
    text = _strip_prompt_echo(text)

    # Discard hallucinations:
    #   1. no Hebrew letters → silence or wrong-script glitch
    #   2. one repeated word OR multiple prompt echoes filling the output
    if not _has_hebrew(text) or _looks_like_hallucination(text):
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
    print(f"Text:     {result['text']}")
