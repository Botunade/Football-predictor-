@echo off
echo Multi-Sport Predictor V4 Web Interface
echo =======================================
echo Installing web dependencies...
pip install -r requirements_web.txt -q
pip install -r requirements.txt -q
echo.
echo Starting web server at http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.
python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); import os; os.makedirs('src/web/templates', exist_ok=True); from src.web.app import create_templates; create_templates()"
python run.py