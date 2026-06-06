"""
Guard Call — FastAPI Backend
"""
import asyncio
import base64
import json
import os
import tempfile
import time
from datetime import datetime

import aiosqlite
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load .env BEFORE importing notifier — it reads env vars on import.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np

try:
    from .analysis import analyze_voice, analyze_waveform, decode_audio
    from .notifier import send_fraud_alert as _send_alert_async
except ImportError:
    from analysis import analyze_voice, analyze_waveform, decode_audio
    from notifier import send_fraud_alert as _send_alert_async

# How many seconds of audio to keep in the rolling per-call buffer fed to the
# voice model. The model needs more than a bare 3 s chunk to be reliable.
VOICE_BUFFER_SEC = 4.0

NLP_SERVICE_URL = "http://localhost:8001"

# Попытка подключить NLP-модули напрямую (резерв если порт 8001 не запущен)
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "text_fraud_recognition"))
try:
    from transcriber import transcribe_audio as _transcribe_audio
    from fraud_detector import check_fraud_phrases as _check_fraud_phrases
    _NLP_DIRECT = True
except Exception:
    _NLP_DIRECT = False

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "calls.db")
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="Guard Call — Audio Fraud Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (logo, etc.) referenced by the frontend at /static/...
STATIC_PATH = os.path.join(FRONTEND_PATH, "static")
os.makedirs(STATIC_PATH, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

async def _nlp_direct(audio_base64: str) -> dict:
    """Вызов NLP-функций напрямую (без HTTP) — резерв когда порт 8001 не запущен."""
    audio_bytes = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(None, _transcribe_audio, tmp_path)
        text = transcription["text"]
        language = transcription["language"]
        fraud = await loop.run_in_executor(None, _check_fraud_phrases, text, language)
        return {
            "text_score": fraud["text_score"],
            "transcript": text,
            "language": language,
            "matched_phrases": fraud["matched_phrases"],
            "category": fraud["category"],
        }
    except Exception:
        return {"text_score": 0.0, "transcript": "", "language": "unknown",
                "matched_phrases": [], "category": "none"}
    finally:
        os.unlink(tmp_path)


async def call_nlp_service(audio_base64: str) -> dict:
    """
    Сначала пробуем HTTP-сервис на порту 8001.
    Если не запущен — вызываем NLP-функции напрямую.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{NLP_SERVICE_URL}/analyze_text",
                json={"audio_base64": audio_base64, "file_extension": "webm"},
            )
            return response.json()
    except Exception:
        if _NLP_DIRECT:
            return await _nlp_direct(audio_base64)
        return {"text_score": 0.0, "transcript": "", "language": "unknown",
                "matched_phrases": [], "category": "none"}


# Shared alert state per active WebSocket connection
_alert_states: dict = {}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                duration_sec    INTEGER NOT NULL,
                max_score       REAL    NOT NULL,
                level           TEXT    NOT NULL,
                matched_phrases TEXT    NOT NULL,
                transcript      TEXT    NOT NULL
            )
        """)
        await db.commit()


async def save_call(duration_sec: int, max_score: float, level: str,
                    matched_phrases: list, transcript: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO calls
               (timestamp, duration_sec, max_score, level, matched_phrases, transcript)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                duration_sec,
                round(max_score, 4),
                level,
                json.dumps(matched_phrases, ensure_ascii=False),
                transcript,
            ),
        )
        await db.commit()


@app.on_event("startup")
async def startup():
    await init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_threat_level(score: float) -> str:
    if score > 0.8:
        return "alert"
    elif score > 0.6:
        return "danger"
    elif score >= 0.4:
        return "warning"
    return "safe"


def send_fraud_alert(matched_phrases: list, transcript: str):
    """Fire WhatsApp alert via Twilio (notifier.py). Non-blocking."""
    print(f"[FRAUD ALERT] Phrases: {matched_phrases} | '{transcript[:80]}'")
    # Schedule async send without blocking the WebSocket loop.
    asyncio.create_task(_send_alert_async(matched_phrases, transcript))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/history")
async def get_history():
    """Return last 20 call sessions from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM calls ORDER BY id DESC LIMIT 20"
        ) as cursor:
            rows = await cursor.fetchall()

    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "duration_sec": row["duration_sec"],
            "max_score": row["max_score"],
            "level": row["level"],
            "matched_phrases": json.loads(row["matched_phrases"]),
            "transcript": row["transcript"],
        }
        for row in rows
    ]


@app.post("/test_alert")
async def test_alert():
    """Manual trigger for the WhatsApp alert — bypasses the score gate.
    Use curl -X POST http://localhost:8000/test_alert to debug Twilio."""
    fake_phrases = ["מהבנק שלך נפרץ", "אשר את מספר הכרטיס"]
    fake_transcript = "שלום, אני מהבנק. החשבון שלך נפרץ."
    try:
        await _send_alert_async(fake_phrases, fake_transcript)
        return {"status": "alert dispatched — check terminal logs and your WhatsApp"}
    except Exception as e:
        return {"status": "error", "detail": f"{type(e).__name__}: {e}"}


@app.post("/cancel_alert")
async def cancel_alert():
    """Reset alert counter for all active connections (false-positive button)."""
    for state in _alert_states.values():
        state["consecutive_high"] = 0
        state["alert_sent"] = False
    return {"status": "alert cancelled"}

def normalize_1(num: float) -> float:
    if num < 0.2:
        return num
    elif num < 0.4:
        return (num-0.2)*0.8/0.3+0.2
    else:
        return (num-0.5)*0.2/0.5+0.8


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Accepts base64-encoded audio every 3 seconds.
    Returns JSON:
    {
        "voice_score": 0.72,
        "text_score": 0.60,
        "final_score": 0.67,
        "level": "warning",         # safe / warning / danger / alert
        "matched_phrases": [...],
        "transcript": "...",
        "language": "he",
        "alert_sent": false
    }
    """
    await websocket.accept()

    conn_id = id(websocket)
    state = {
        "consecutive_high": 0,
        "alert_sent": False,
        "vbuf": np.zeros(0, dtype=np.float32),  # rolling decoded audio
        "vsr": 16000,
        "voice_ema": None,                       # smoothed voice score
    }
    _alert_states[conn_id] = state

    session_start = time.time()
    session_max_score = 0.0
    session_transcript = ""
    session_phrases: list = []

    try:
        while True:
            data = await websocket.receive_text()

            # Decode base64 audio to a temporary file
            try:
                audio_bytes = base64.b64decode(data)
            except Exception:
                audio_bytes = data.encode()

            # Calibration capture: if CAPTURE_DIR is set, dump every incoming
            # chunk (exactly what the model will see live) to that folder.
            # Usage:  CAPTURE_DIR=data/human ./start.sh   -> talk, then Ctrl+C.
            _cap = os.environ.get("CAPTURE_DIR")
            if _cap:
                os.makedirs(_cap, exist_ok=True)
                with open(os.path.join(_cap, f"live_{int(time.time()*1000)}.wav"), "wb") as _cf:
                    _cf.write(audio_bytes)

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Decode this chunk and append to the rolling buffer so the model
            # sees ~VOICE_BUFFER_SEC of context, not a bare 3 s slice.
            try:
                cwav, csr = decode_audio(tmp_path)
                if csr != state["vsr"]:
                    state["vbuf"] = np.zeros(0, dtype=np.float32)
                    state["vsr"] = csr
                state["vbuf"] = np.concatenate([state["vbuf"], cwav])
                maxlen = int(csr * VOICE_BUFFER_SEC)
                if state["vbuf"].size > maxlen:
                    state["vbuf"] = state["vbuf"][-maxlen:]
                voice_coro = analyze_waveform(state["vbuf"], state["vsr"])
            except Exception:
                voice_coro = analyze_voice(tmp_path)

            try:
                voice_result, text_result = await asyncio.gather(
                    voice_coro,                     # local: Conformer model (or heuristic)
                    call_nlp_service(data),         # Participant 2: Whisper + fraud phrases
                )
            finally:
                os.unlink(tmp_path)

            # Raw model/heuristic score (1 = AI). No artificial inflation.
            voice_raw = float(voice_result.get("score", 0.0))
            # Temporal smoothing: one noisy 3 s window should not trip an alert.
            prev = state.get("voice_ema")
            voice_ema = voice_raw if prev is None else 0.6 * prev + 0.4 * voice_raw
            state["voice_ema"] = voice_ema
            voice_score = round(voice_ema, 4)

            text_score = float(text_result.get("text_score", 0.0))
            final_score = round(voice_score * 0.6 + text_score * 0.4, 4)
            level = get_threat_level(final_score)

            matched = text_result.get("matched_phrases", [])
            transcript = text_result.get("transcript", "")
            language = text_result.get("language", "unknown")
            category = text_result.get("category", "none")

            # Update session stats
            session_max_score = max(session_max_score, final_score)
            if transcript:
                session_transcript = transcript
            for p in matched:
                if p not in session_phrases:
                    session_phrases.append(p)

            # Alert logic: trigger after 2 consecutive windows above 0.9
            if final_score > 0.6:
                state["consecutive_high"] += 1
            else:
                state["consecutive_high"] = 0

            if state["consecutive_high"] >= 1 and not state["alert_sent"]:
                state["alert_sent"] = True
                #send_fraud_alert(matched, transcript)

            await websocket.send_json({
                "voice_score": voice_score,
                "text_score": text_score,
                "final_score": final_score,
                "level": level,
                "matched_phrases": matched,
                "transcript": transcript,
                "language": language,
                "category": category,
                "alert_sent": state["alert_sent"],
            })

    except WebSocketDisconnect:
        pass
    finally:
        _alert_states.pop(conn_id, None)
        duration = int(time.time() - session_start)
        await save_call(
            duration,
            session_max_score,
            get_threat_level(session_max_score),
            session_phrases,
            session_transcript,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
