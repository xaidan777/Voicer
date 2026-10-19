@echo off
chcp 65001 >nul
rem Voicer: веб-интерфейс http://127.0.0.1:8765 (откроется в браузере). Первый запуск сам всё установит.
cd /d "%~dp0"
if not exist ".venv\voicer-ok" (
  call "%~dp0install.bat" auto
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m voicer serve %*
pause
