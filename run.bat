@echo off
title RestoreAI Web Server
echo Starting RestoreAI...
cd /d "%~dp0"
call venv\Scripts\python.exe app.py
pause
