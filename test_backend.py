"""
Guard Call — быстрый тест бэкенда
Запуск: python3 test_backend.py
"""
import asyncio
import base64
import json
import subprocess
import sys
import time

import httpx

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/stream"

# Аудио файлы для теста (из папки text_fraud_recognition/)
AUDIO_AI    = "text_fraud_recognition/hebrew_3_ai.mp3"       # AI-голос → должен дать высокий скор
AUDIO_HUMAN = "text_fraud_recognition/test_call_human_he.m4a" # Живой голос → должен дать низкий скор


def print_result(label: str, data: dict):
    score = data.get("final_score", data.get("text_score", "?"))
    level = data.get("level", "—")
    transcript = data.get("transcript", "—")
    phrases = data.get("matched_phrases", [])
    category = data.get("category", "—")

    colors = {"safe": "\033[92m", "warning": "\033[93m",
              "danger": "\033[91m", "alert": "\033[91m"}
    reset = "\033[0m"
    color = colors.get(level, "")

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Итоговый скор:  {color}{round(float(score)*100) if score != '?' else '?'}%{reset}")
    print(f"  Уровень:        {color}{level.upper()}{reset}")
    if transcript:
        print(f"  Транскрипт:     {transcript[:80]}")
    if phrases:
        print(f"  ⚠️  Фразы:       {', '.join(phrases[:3])}")
    if category and category != "none":
        print(f"  Категория:      {category}")
    print()


async def test_health():
    print("1️⃣  Проверка /health ...")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BACKEND_URL}/health")
        assert r.status_code == 200, f"Бэкенд не отвечает: {r.status_code}"
        print("   ✅ Бэкенд работает")


async def test_history():
    print("2️⃣  Проверка /history ...")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BACKEND_URL}/history")
        assert r.status_code == 200
        data = r.json()
        print(f"   ✅ История доступна ({len(data)} записей)")


async def test_websocket(audio_path: str, label: str):
    import websockets
    print(f"3️⃣  WebSocket тест: {label} ...")
    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        print(f"   ⚠️  Файл не найден: {audio_path} — пропускаю")
        return

    async with websockets.connect(WS_URL) as ws:
        await ws.send(audio_b64)
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        data = json.loads(raw)
        print_result(label, data)


async def run_all_tests():
    print("\n🛡️  Guard Call — тест бэкенда\n")

    # 1. Health check
    try:
        await test_health()
    except Exception as e:
        print(f"   ❌ Бэкенд не запущен: {e}")
        print("   Запусти в другом терминале: uvicorn backend.main:app --port 8000")
        return

    # 2. History
    try:
        await test_history()
    except Exception as e:
        print(f"   ❌ /history ошибка: {e}")

    # 3. WebSocket с аудио файлами
    try:
        await test_websocket(AUDIO_AI,    "AI-голос (ожидаем высокий скор 🔴)")
        await test_websocket(AUDIO_HUMAN, "Живой голос (ожидаем низкий скор 🟢)")
    except ImportError:
        print("   ⚠️  Установи websockets: pip install websockets")
    except Exception as e:
        print(f"   ❌ WebSocket ошибка: {e}")

    print("✅ Тест завершён\n")


async def test_my_audio(audio_path: str):
    print(f"\n🎤  Анализирую твой файл: {audio_path}\n")

    try:
        await test_health()
    except Exception:
        print("❌ Бэкенд не запущен. Сначала запусти: uvicorn backend.main:app --port 8000")
        return

    await test_websocket(audio_path, f"Твой файл: {audio_path.split('/')[-1]}")
    print("✅ Готово\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # python3 test_backend.py /path/to/audio.mp3
        asyncio.run(test_my_audio(sys.argv[1]))
    else:
        asyncio.run(run_all_tests())
