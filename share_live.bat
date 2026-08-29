@echo off
title BORDER AI — Live Sharing & Testing
echo ========================================================
echo   BORDER AI: Public Testing Server
echo   Smart India Hackathon 2026
echo ========================================================
echo.
echo Starting Local AI Server...
start "BORDER AI Server" py -3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
timeout /t 3 /nobreak >nul

echo Starting Public HTTPS Tunnel...
echo Copy the trycloudflare.com link below and share it with your friends!
echo ========================================================
echo.
"%~dp0cloudflared.exe" tunnel --url http://127.0.0.1:8000

pause
