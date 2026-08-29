@echo off
title BORDER AI — SIH 2026 Surveillance System
echo ========================================================
echo   BORDER AI: Intelligent Video Analytics for Surveillance
echo   Smart India Hackathon 2026
echo ========================================================
echo.
echo Starting BORDER AI Server on http://127.0.0.1:8000 ...
echo Press Ctrl+C anytime to stop.
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
