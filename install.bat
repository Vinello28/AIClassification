@echo off
echo.
echo ========================================
echo   Classificatore AI - Setup Automatico
echo ========================================
echo.

REM Controlla che Python sia installato
python --version >nul 2>&1
if errorlevel 1 (
    echo Errore: Python non trovato nel PATH
    echo Installa Python 3.8+ da https://python.org
    pause
    exit /b 1
)

echo Python trovato, avvio setup...
echo.

REM Esegue lo script di setup Python
python setup.py

echo.
echo Setup completato!
echo.
echo Per testare il classificatore:
echo   python example_usage.py
echo.
pause
