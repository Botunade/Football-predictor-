"""
Multi-Sport Predictor V4 - Local Server Launcher
Provides foolproof local execution with full Windows Proactor loop support.
"""
import sys
import os
import asyncio

# Ensure UTF-8 output on Windows PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # Enforce Windows Proactor Loop for Playwright subprocess compatibility
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.utils.proactor_helper import patch_windows_asyncio
patch_windows_asyncio()

if __name__ == "__main__":
    import uvicorn
    from src.web.app import create_templates
    create_templates()

    print("\n" + "=" * 65)
    print("  🚀 MULTI-SPORT PREDICTOR V4 - LOCAL ENGINE RUNNING")
    print("  🌐 Access the Dashboard at: http://127.0.0.1:8000")
    print("  ⚡ Windows Proactor EventLoop: ACTIVE (Playwright ready)")
    print("  🔒 Press Ctrl+C in this PowerShell window to stop")
    print("=" * 65 + "\n")

    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=False)
