@echo off
chcp 65001 >nul
rem Установка Voicer: uv, Python 3.11 и библиотеки в папку .venv. Повторный запуск обновляет библиотеки.
rem start.bat вызывает этот файл сам при первом запуске.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

where uv >nul 2>nul
if errorlevel 1 (
  echo Устанавливаю uv — менеджер Python: https://docs.astral.sh/uv/
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo Не удалось установить uv. Поставьте его вручную: https://docs.astral.sh/uv/
  goto fail
)

if not exist "%PY%" (
  uv venv .venv --python 3.11
  if errorlevel 1 goto fail
)

echo Библиотеки: распознавание речи, обработка звука, веб-интерфейс...
uv pip install --python "%PY%" --no-deps -r requirements-win.txt
if errorlevel 1 goto fail
uv pip install --python "%PY%" --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu torchaudio==2.11.0+cpu
if errorlevel 1 goto fail
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo Найдена видеокарта NVIDIA: ставлю библиотеки CUDA для быстрого распознавания
  uv pip install --python "%PY%" --no-deps -r requirements-gpu.txt
  if errorlevel 1 goto fail
)
uv pip install --python "%PY%" --no-deps -e .
if errorlevel 1 goto fail

echo ok> .venv\voicer-ok
echo.
echo Готово. Запуск: start.bat
echo Модель распознавания (~1,6 ГБ с видеокартой, ~0,5 ГБ без) скачается при первой записи.
if "%~1"=="" pause
exit /b 0

:fail
echo.
echo Установка не удалась: причина выше. Запустите install.bat ещё раз или напишите в Issues на GitHub.
if "%~1"=="" pause
exit /b 1
