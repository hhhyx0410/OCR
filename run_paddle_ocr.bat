@echo off
cd /d "%~dp0"

if exist "%~dp0ocr_images" (
    set "INPUT_PATH=%~dp0ocr_images"
) else (
    set "INPUT_PATH=%~dp0"
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%~dp0paddle_ocr_local.py" "%INPUT_PATH%"
    goto end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0paddle_ocr_local.py" "%INPUT_PATH%"
    goto end
)

echo Python not found. Please install Python 3.10+ and make sure py or python is available.

:end
pause
