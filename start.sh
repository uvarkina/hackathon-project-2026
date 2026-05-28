#!/bin/bash
# Guard Call — запуск всего проекта

cd "$(dirname "$0")"

echo "=== Guard Call ==="
echo "Устанавливаю зависимости..."
pip install -r requirements.txt -q

echo "Запускаю бэкенд на порту 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo ""
echo "Бэкенд запущен (PID $BACKEND_PID)"
echo "API:      http://localhost:8000"
echo "История:  http://localhost:8000/history"
echo "Здоровье: http://localhost:8000/health"
echo ""

sleep 2
# Открыть фронтенд в браузере (Mac)
open frontend/index.html 2>/dev/null || true

echo "Нажми Ctrl+C для остановки."
wait $BACKEND_PID
