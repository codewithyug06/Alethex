"""
ALETHEX Master Automated Launcher.
One single script that automatically:
1. Scans and auto-attaches ALETHEX to all detected AI apps (Claude Desktop, Cursor, Ollama, LM Studio).
2. Launches background Universal Memory Gateway (port 8000) & RAG microservice (port 8080).
3. Opens the Landing Page & Live Demo in the user's default browser.
4. Starts the Desktop Companion application.
"""

import sys
import os
import subprocess
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from alethex.integrations.auto_attach import AIAttachDetector


def main():
    print("=" * 65)
    print("   ALETHEX: Universal AI Memory & Truth Engine - Auto Launch")
    print("=" * 65)

    # 1. Auto-Scan & Auto-Attach
    print("\n[STEP 1/4] Scanning local system for installed AI applications...")
    detector = AIAttachDetector()
    scans = detector.scan_all()
    for s in scans:
        status_tag = "[DETECTED]" if s["installed"] else "[--]"
        print(f"  {status_tag} {s['name']}: {s['status']}")

    print("\n[STEP 2/4] Auto-attaching ALETHEX to all detected tools...")
    attached = detector.attach_all()
    for k, v in attached.items():
        print(f"  --> {k}: {v}")

    # 2. Launch Universal RAG microservice in background
    print("\n[STEP 3/4] Starting background Universal RAG microservice on http://localhost:8080...")
    python_exe = sys.executable
    rag_cmd = [python_exe, "-m", "alethex.integrations.rag_service", "--port", "8080"]
    rag_proc = subprocess.Popen(rag_cmd, cwd=str(PROJECT_ROOT))
    time.sleep(1.5)

    # 3. Open Landing Page in Browser
    print("\n[STEP 4/4] Opening interactive Landing Page in default browser...")
    landing_file = PROJECT_ROOT / "alethex" / "reporting" / "landing_page.html"
    if landing_file.exists():
        webbrowser.open(landing_file.as_uri())

    print("\n" + "=" * 65)
    print("   ALETHEX is ACTIVE and running!")
    print("   - RAG Microservice: http://localhost:8080")
    print("   - Extension Package: dist/alethex-extension-v1.0.0.zip")
    print("   - Desktop Companion: alethex_desktop.py")
    print("=" * 65 + "\n")

    # Start Desktop Companion GUI
    try:
        from alethex_desktop import main as run_gui
        run_gui()
    except Exception as e:
        print(f"Desktop GUI closed or running in headless mode: {e}")
        try:
            rag_proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down ALETHEX...")
            rag_proc.terminate()


if __name__ == "__main__":
    main()
