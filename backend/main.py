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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from .analysis import analyze_voice
except ImportError:
    from analysis import analyze_voice

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

app = FastAPI(title="Guard Call — Audio Fraud Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _nlp_direct(audio_base64: str) -> dict:
    """Вызов NLP-функций напрямую (без HTTP) — резерв когда порт 8001 не запущен."""
    audio_bytes = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
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


_DEFAULT_RESULT = {"text_score": 0.0, "transcript": "", "language": "unknown",
                   "matched_phrases": [], "category": "none"}

async def call_nlp_service(audio_base64: str) -> dict:
    """
    Сначала пробуем HTTP-сервис на порту 8001.
    Если не запущен — вызываем NLP-функции напрямую.
    Все ошибки поглощаются — возвращаем дефолт вместо краша.
    """
    # 1. HTTP к NLP-сервису
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{NLP_SERVICE_URL}/analyze_text",
                json={"audio_base64": audio_base64, "file_extension": "wav"},
            )
            if response.status_code == 200:
                return response.json()
            print(f"[NLP HTTP] status {response.status_code}: {response.text[:120]}")
    except Exception as e:
        print(f"[NLP HTTP] unavailable: {e}")

    # 2. Прямой вызов модулей
    if _NLP_DIRECT:
        try:
            return await _nlp_direct(audio_base64)
        except Exception as e:
            print(f"[NLP DIRECT] failed: {e}")

    return dict(_DEFAULT_RESULT)


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
    if score > 0.9:
        return "alert"
    elif score > 0.7:
        return "danger"
    elif score >= 0.4:
        return "warning"
    return "safe"


def send_fraud_alert(matched_phrases: list, transcript: str):
    """Stub — Participant 5 replaces this with Twilio."""
    print(f"[FRAUD ALERT] Phrases: {matched_phrases} | '{transcript[:80]}'")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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


@app.post("/cancel_alert")
async def cancel_alert():
    """Reset alert counter for all active connections (false-positive button)."""
    for state in _alert_states.values():
        state["consecutive_high"] = 0
        state["alert_sent"] = False
    return {"status": "alert cancelled"}


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
    state = {"consecutive_high": 0, "alert_sent": False}
    _alert_states[conn_id] = state

    session_start = time.time()
    session_max_score = 0.0
    session_transcript = ""
    session_phrases: list = []
    session_hits = 0  # суммарное кол-во срабатываний (включая повторы)

    try:
        while True:
            data = await websocket.receive_text()

            # Decode base64 audio to a temporary file
            try:
                audio_bytes = base64.b64decode(data)
            except Exception:
                audio_bytes = data.encode()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                text_result = await call_nlp_service(data)
            except Exception as e:
                print(f"[WS] call_nlp_service unexpected error: {e}")
                text_result = dict(_DEFAULT_RESULT)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            text_score = float(text_result.get("text_score", 0.0))
            transcript = text_result.get("transcript", "")

            # Накапливаем фразы и суммарные срабатывания
            matched = text_result.get("matched_phrases", [])
            for p in matched:
                if p not in session_phrases:
                    session_phrases.append(p)
            session_hits += len(matched)  # каждое срабатывание считается

            # Session score: растёт по hits (каждое упоминание фразы = +)
            h = session_hits
            if h == 0:    session_score = 0.0
            elif h == 1:  session_score = 0.40
            elif h == 2:  session_score = 0.52
            elif h == 3:  session_score = 0.62
            elif h == 4:  session_score = 0.70
            elif h == 5:  session_score = 0.76
            elif h == 6:  session_score = 0.82
            elif h == 7:  session_score = 0.87
            elif h == 8:  session_score = 0.91
            elif h == 9:  session_score = 0.94
            else:         session_score = min(0.94 + (h - 9) * 0.01, 0.99)

            final_score = round(max(text_score, session_score), 4)
            print(f"[WS] hits={h} unique={len(session_phrases)} final={final_score} | '{transcript[:50]}'")
            level = get_threat_level(final_score)

            language = text_result.get("language", "unknown")
            category = text_result.get("category", "none")

            # Update session stats
            session_max_score = max(session_max_score, final_score)
            if transcript:
                session_transcript = transcript

            # Alert logic: trigger after 2 consecutive windows above 0.9
            if final_score > 0.9:
                state["consecutive_high"] += 1
            else:
                state["consecutive_high"] = 0

            if state["consecutive_high"] >= 2 and not state["alert_sent"]:
                state["alert_sent"] = True
                send_fraud_alert(matched, transcript)

            await websocket.send_json({
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
    except Exception as e:
        print(f"[WS] handler crashed: {e}")
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
