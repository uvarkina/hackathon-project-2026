"""
Audio Fraud Detection System - FastAPI Backend (Stub Mode)
"""
import asyncio
import json
import random

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Audio Fraud Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Stub phrases ---
FRAUD_PHRASES = ["חשבון הבנק", "אל תגיד לאף אחד", "משטרה"]


# --- Stub analysis functions ---

async def analyze_voice() -> float:
    """Stub: returns random voice fraud score between 0.3 and 0.9."""
    await asyncio.sleep(0.1)  # simulate processing
    return round(random.uniform(0.3, 0.9), 4)


async def analyze_text() -> dict:
    """Stub: returns random text score and a random matched phrase."""
    await asyncio.sleep(0.1)  # simulate processing
    score = round(random.uniform(0.2, 0.8), 4)
    phrase = random.choice(FRAUD_PHRASES)
    return {
        "score": score,
        "matched_phrases": [phrase],
        "transcript": f"...{phrase}...",
    }


def get_threat_level(score: float) -> str:
    """Determine threat level from final score."""
    if score > 0.75:
        return "danger"
    elif score >= 0.4:
        return "warning"
    return "safe"


# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio fraud detection.

    Accepts connection, then every 3 seconds runs voice + text analysis
    in parallel and sends back a JSON result.

    Response format:
    {
        "voice_score": 0.72,
        "text_score": 0.55,
        "final_score": 0.65,
        "level": "warning",
        "matched_phrases": ["חשבון הבנק"],
        "transcript": "...חשבון הבנק..."
    }
    """
    await websocket.accept()

    try:
        while True:
            # Wait for audio chunk from client (base64 string or any message)
            data = await websocket.receive_text()

            # Run voice and text analysis in parallel (stubs)
            voice_score, text_result = await asyncio.gather(
                analyze_voice(),
                analyze_text(),
            )

            text_score = text_result["score"]
            final_score = round(voice_score * 0.6 + text_score * 0.4, 4)

            response = {
                "voice_score": voice_score,
                "text_score": text_score,
                "final_score": final_score,
                "level": get_threat_level(final_score),
                "matched_phrases": text_result["matched_phrases"],
                "transcript": text_result["transcript"],
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
