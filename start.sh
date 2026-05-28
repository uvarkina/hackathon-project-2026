#!/bin/bash
# Guard Call — запуск всего проекта

cd "$(dirname "$0")"

echo "=== Guard Call ==="
echo "Устанавливаю зависимости..."
pip install -r requirements.txt -q

echo ""
echo "Запускаю NLP-сервис (Участник 2) на порту 8001..."
(cd text_fraud_recognition && uvicorn app:app --host 0.0.0.0 --port 8001 --reload) &
NLP_PID=$!

echo "Запускаю основной бэкенд на порту 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Запускаю HTTP-сервер для фронтенда на порту 3001..."
python3 -m http.server 3001 --directory frontend &
FRONTEND_PID=$!

echo ""
echo "Сервисы запущены:"
echo "  Фронтенд:           http://localhost:3001"
echo "  Бэкенд (WebSocket): http://localhost:8000"
echo "  NLP-сервис:         http://localhost:8001"
echo "  История звонков:    http://localhost:8000/history"
echo ""
echo "Для ngrok-демо: передай URL через параметр ?server=wss://xxx.ngrok.io/ws/stream"
echo ""

sleep 2
open "http://localhost:3001" 2>/dev/null || true

echo "Нажми Ctrl+C для остановки."

trap "kill $BACKEND_PID $NLP_PID $FRONTEND_PID 2>/dev/null" EXIT
wait $BACKEND_PID
