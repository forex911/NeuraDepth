@echo off
echo =========================================
echo Starting Depth Scanner Services...
echo =========================================

echo.
echo [1/2] Starting Python Backend...
start "Depth Scanner Backend" cmd /k "cd backend && call .venv\Scripts\activate && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

echo [2/2] Starting React Frontend...
start "Depth Scanner Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both services are starting up in separate windows!
echo You can close this window now. The services will continue running.
echo To stop the servers, close their respective command windows.
