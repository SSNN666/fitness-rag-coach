@echo off
rem ============================================================
rem  One-click start: API backend + Streamlit UI
rem  Usage: double-click this file, or:
rem    start.bat --no-browser    (no auto browser)
rem    start.bat --skip-api      (UI only, graph tab works)
rem  Ctrl+C stops both services.
rem ============================================================
cd /d %~dp0

rem Prefer the project virtualenv; fall back to PATH python
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% -u start.py %*
if errorlevel 1 pause
