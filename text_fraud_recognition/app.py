import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from transcriber import transcribe_audio
from fraud_detector import check_fraud_phrases

app = FastAPI(title="Fraud Call Detector API")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".wma", ".aac", ".opus"}


class AudioBase64Request(BaseModel):
    audio_base64: str
    file_extension: str = "wav"


class AnalyzeResponse(BaseModel):
    text_score: float
    transcript: str
    language: str
    matched_phrases: list[str]
    category: str
    pattern_score: float
    ai_score: float


def _analyze_audio_file(file_path: str) -> AnalyzeResponse:
    """Shared logic: transcribe + fraud check."""
    transcription = transcribe_audio(file_path)
    transcript_text = transcription["text"]
    language = transcription["language"]

    fraud_result = check_fraud_phrases(transcript_text, language)

    return AnalyzeResponse(
        text_score=fraud_result["text_score"],
        transcript=transcript_text,
        language=language,
        matched_phrases=fraud_result["matched_phrases"],
        category=fraud_result["category"],
        pattern_score=fraud_result["pattern_score"],
        ai_score=fraud_result["ai_score"],
    )


@app.post("/analyze_text", response_model=AnalyzeResponse)
async def analyze_text_base64(request: AudioBase64Request):
    """
    Accept base64-encoded audio, transcribe it, and check for fraud phrases.
    Supports any audio format (wav, mp3, ogg, flac, m4a, webm, etc.)
    """
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    suffix = f".{request.file_extension.lstrip('.')}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return _analyze_audio_file(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/analyze_file", response_model=AnalyzeResponse)
async def analyze_file_upload(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file directly (mp3, wav, ogg, flac, m4a, webm, etc.)

    Usage with curl:
        curl -X POST http://localhost:8000/analyze_file -F "file=@call.mp3"
    """
    original_name = file.filename or "audio.wav"
    suffix = Path(original_name).suffix or ".wav"

    if suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return _analyze_audio_file(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
