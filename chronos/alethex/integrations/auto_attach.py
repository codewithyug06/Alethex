"""
ALETHEX Universal Auto-Detector & Auto-Attach Engine.
Automatically scans a user's operating system (Windows / macOS / Linux)
for installed AI applications and automatically wires up ALETHEX in 1 click:

Supported AI Systems:
1. Claude Desktop (auto-injects MCP into claude_desktop_config.json)
2. Cursor IDE (auto-injects MCP into ~/.cursor/mcp.json)
3. Windsurf IDE (auto-injects MCP into ~/.codeium/windsurf/mcp_config.json)
4. Ollama (detects binary + port 11434, exposes reconciled Modelfile & proxy)
5. LM Studio (detects port 1234)
6. Open-WebUI (detects standard Docker / local ports)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

logger = logging.getLogger("alethex-auto-attach")


class AIAttachDetector:
    def __init__(self):
        self.home = Path.home()
        self.is_win = sys.platform.startswith("win")
        self.is_mac = sys.platform == "darwin"
        self.is_linux = sys.platform.startswith("linux")

    def detect_claude_desktop(self) -> Dict[str, Any]:
        """Checks for Claude Desktop config location."""
        if self.is_win:
            appdata = os.environ.get("APPDATA")
            cfg_path = Path(appdata) / "Claude" / "claude_desktop_config.json" if appdata else self.home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        elif self.is_mac:
            cfg_path = self.home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        else:
            cfg_path = self.home / ".config" / "Claude" / "claude_desktop_config.json"

        installed = cfg_path.parent.exists() or cfg_path.exists()
        return {
            "name": "Claude Desktop",
            "installed": installed,
            "config_path": str(cfg_path),
            "status": "Ready to attach" if installed else "Not detected"
        }

    def detect_cursor(self) -> Dict[str, Any]:
        """Checks for Cursor IDE MCP config."""
        cursor_dir = self.home / ".cursor"
        cfg_path = cursor_dir / "mcp.json"
        installed = cursor_dir.exists()
        return {
            "name": "Cursor IDE",
            "installed": installed,
            "config_path": str(cfg_path),
            "status": "Ready to attach" if installed else "Not detected"
        }

    def detect_windsurf(self) -> Dict[str, Any]:
        """Checks for Windsurf IDE config."""
        windsurf_dir = self.home / ".codeium" / "windsurf"
        cfg_path = windsurf_dir / "mcp_config.json"
        installed = windsurf_dir.exists()
        return {
            "name": "Windsurf IDE",
            "installed": installed,
            "config_path": str(cfg_path),
            "status": "Ready to attach" if installed else "Not detected"
        }

    def detect_ollama(self) -> Dict[str, Any]:
        """Checks for local Ollama installation & running server."""
        ollama_bin = False
        running = False

        if self.is_win:
            local_appdata = os.environ.get("LOCALAPPDATA")
            if local_appdata and (Path(local_appdata) / "Programs" / "Ollama" / "ollama.exe").exists():
                ollama_bin = True
        
        # Test port 11434
        try:
            req = urllib.request.Request("http://localhost:11434/api/version", headers={"User-Agent": "ALETHEX"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    running = True
                    ollama_bin = True
        except Exception:
            pass

        return {
            "name": "Ollama (Local LLMs)",
            "installed": ollama_bin or running,
            "running": running,
            "endpoint": "http://localhost:11434",
            "status": "Server Active" if running else ("Installed (Offline)" if ollama_bin else "Not detected")
        }

    def detect_lm_studio(self) -> Dict[str, Any]:
        """Checks for LM Studio local inference server."""
        running = False
        try:
            req = urllib.request.Request("http://localhost:1234/v1/models", headers={"User-Agent": "ALETHEX"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    running = True
        except Exception:
            pass

        return {
            "name": "LM Studio",
            "installed": running,
            "endpoint": "http://localhost:1234/v1",
            "status": "Server Active" if running else "Not detected"
        }

    def scan_all(self) -> List[Dict[str, Any]]:
        return [
            self.detect_claude_desktop(),
            self.detect_cursor(),
            self.detect_windsurf(),
            self.detect_ollama(),
            self.detect_lm_studio(),
        ]

    def attach_mcp_config(self, target_cfg_path: Path, python_exe: Optional[str] = None) -> bool:
        """Injects ALETHEX MCP server into any Claude / Cursor / Windsurf config file."""
        python_bin = python_exe or sys.executable
        target_cfg_path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {}
        if target_cfg_path.exists():
            try:
                with open(target_cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if "mcpServers" not in data:
            data["mcpServers"] = {}

        data["mcpServers"]["alethex"] = {
            "command": python_bin,
            "args": ["-m", "alethex.integrations.mcp_server"]
        }

        with open(target_cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True

    def attach_all(self) -> Dict[str, Any]:
        """Automatically wires up all detected tools."""
        results = {}
        # 1. Claude Desktop
        claude_info = self.detect_claude_desktop()
        cfg_p = Path(claude_info["config_path"])
        self.attach_mcp_config(cfg_p)
        results["Claude Desktop"] = f"Attached successfully to {cfg_p}"

        # 2. Cursor
        cursor_info = self.detect_cursor()
        if cursor_info["installed"]:
            cur_p = Path(cursor_info["config_path"])
            self.attach_mcp_config(cur_p)
            results["Cursor"] = f"Attached successfully to {cur_p}"

        # 3. Windsurf
        windsurf_info = self.detect_windsurf()
        if windsurf_info["installed"]:
            ws_p = Path(windsurf_info["config_path"])
            self.attach_mcp_config(ws_p)
            results["Windsurf"] = f"Attached successfully to {ws_p}"

        # 4. Ollama
        ollama_info = self.detect_ollama()
        if ollama_info["installed"]:
            results["Ollama"] = "Configured: Run 'alethex proxy --upstream http://localhost:11434/v1' to route all Ollama queries through ALETHEX"

        return results


def main():
    detector = AIAttachDetector()
    print("\n[SCAN] Scanning system for installed AI applications...")
    scans = detector.scan_all()
    for s in scans:
        status_tag = "[DETECTED]" if s["installed"] else "[--]"
        print(f"  {status_tag} {s['name']}: {s['status']}")

    print("\n[ATTACH] Auto-attaching ALETHEX to detected applications...")
    attached = detector.attach_all()
    for k, v in attached.items():
        print(f"  --> {k}: {v}")


if __name__ == "__main__":
    main()

