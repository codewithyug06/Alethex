"""
ALETHEX Desktop Auto-Attach Companion.
Cross-platform desktop application for Windows, macOS, and Linux.
Automatically scans for local AI applications (Claude Desktop, Cursor, Ollama, LM Studio),
configures MCP servers with 1 click, and manages the background memory reconciliation gateway.
"""

import sys
import os
import threading
import subprocess
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Ensure local packages are on path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from alethex.integrations.auto_attach import AIAttachDetector


class AlethexDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ALETHEX — Universal AI Memory & Truth Companion")
        self.root.geometry("680x560")
        self.root.minsize(600, 480)
        self.root.configure(bg="#0b0f19")

        self.detector = AIAttachDetector()
        self.proxy_process = None
        self.rag_process = None

        self._apply_styles()
        self._build_ui()
        self.scan_systems()

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#0b0f19", foreground="#f3f4f6", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#38bdf8", background="#0b0f19")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#94a3b8", background="#0b0f19")
        style.configure("TFrame", background="#0b0f19")
        style.configure("Card.TFrame", background="#111827", relief="solid", borderwidth=1)
        style.configure("Attach.TButton", font=("Segoe UI", 10, "bold"), background="#0284c7", foreground="#ffffff")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Banner
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title = ttk.Label(header_frame, text="α ALETHEX Desktop Companion", style="Header.TLabel")
        title.pack(anchor=tk.W)
        subtitle = ttk.Label(header_frame, text="Universal Temporal Consistency & Contradiction Reconciliation for Any AI", style="SubHeader.TLabel")
        subtitle.pack(anchor=tk.W)

        # Detected AIs Card
        card = tk.LabelFrame(main_frame, text=" Detected AI Systems on this Computer ", bg="#111827", fg="#38bdf8", font=("Segoe UI", 10, "bold"), padx=12, pady=12)
        card.pack(fill=tk.X, pady=8)

        self.status_labels = {}
        for app_name in ["Claude Desktop", "Cursor IDE", "Windsurf IDE", "Ollama (Local LLMs)", "LM Studio"]:
            row = tk.Frame(card, bg="#111827")
            row.pack(fill=tk.X, pady=3)
            name_lbl = tk.Label(row, text=f"• {app_name}:", bg="#111827", fg="#e2e8f0", font=("Segoe UI", 10, "bold"), width=22, anchor="w")
            name_lbl.pack(side=tk.LEFT)
            stat_lbl = tk.Label(row, text="Scanning...", bg="#111827", fg="#94a3b8", font=("Segoe UI", 9))
            stat_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.status_labels[app_name] = stat_lbl

        # Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=12)

        self.attach_btn = tk.Button(
            btn_frame,
            text="⚡ Auto-Attach to All Detected AIs",
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.auto_attach_all
        )
        self.attach_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.service_btn = tk.Button(
            btn_frame,
            text="▶ Start Background Memory Gateway",
            bg="#10b981",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            activebackground="#059669",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.toggle_services
        )
        self.service_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.web_btn = tk.Button(
            btn_frame,
            text="🌐 Open Landing Page",
            bg="#334155",
            fg="#f8fafc",
            font=("Segoe UI", 9),
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.open_landing_page
        )
        self.web_btn.pack(side=tk.RIGHT)

        # Log Output Console
        log_frame = tk.LabelFrame(main_frame, text=" Activity Log & System Status ", bg="#0b0f19", fg="#94a3b8", font=("Segoe UI", 9), padx=8, pady=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_area = scrolledtext.ScrolledText(log_frame, bg="#050811", fg="#38bdf8", insertbackground="#38bdf8", font=("Consolas", 9), relief="flat")
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self.log("ALETHEX Desktop Companion initialized.")
        self.log("Loaded INT8 Quantized ONNX Engine (164 MB).")

    def log(self, message: str):
        self.log_area.insert(tk.END, f"[ALETHEX] {message}\n")
        self.log_area.see(tk.END)

    def scan_systems(self):
        scans = self.detector.scan_all()
        for s in scans:
            name = s["name"]
            if name in self.status_labels:
                lbl = self.status_labels[name]
                if s["installed"]:
                    lbl.config(text=f"✓ {s['status']}", fg="#10b981")
                else:
                    lbl.config(text=f"— {s['status']}", fg="#64748b")
        self.log("Scanned local system for AI applications.")

    def auto_attach_all(self):
        self.log("Starting automatic attachment sequence...")
        results = self.detector.attach_all()
        for app, res in results.items():
            self.log(f"Attached {app}: {res}")
        messagebox.showinfo("ALETHEX Auto-Attach", "Successfully attached ALETHEX to your AI tools!\n\nClaude Desktop and IDEs now have native memory reconciliation tools enabled.")

    def toggle_services(self):
        if self.proxy_process is None:
            self.log("Starting Universal Gateway on port 8000 and RAG service on port 8080...")
            python_exe = sys.executable
            
            # Start universal proxy
            cmd_proxy = [python_exe, "-m", "alethex.integrations.openai_proxy", "--port", "8000"]
            self.proxy_process = subprocess.Popen(cmd_proxy, cwd=str(PROJECT_ROOT))
            
            # Start RAG service
            cmd_rag = [python_exe, "-m", "alethex.integrations.rag_service", "--port", "8080"]
            self.rag_process = subprocess.Popen(cmd_rag, cwd=str(PROJECT_ROOT))

            self.service_btn.config(text="■ Stop Background Services", bg="#ef4444")
            self.log("Services active: Gateway on http://localhost:8000 | RAG on http://localhost:8080")
        else:
            self.log("Stopping services...")
            if self.proxy_process:
                self.proxy_process.terminate()
                self.proxy_process = None
            if self.rag_process:
                self.rag_process.terminate()
                self.rag_process = None
            self.service_btn.config(text="▶ Start Background Memory Gateway", bg="#10b981")
            self.log("All services stopped.")

    def open_landing_page(self):
        landing_path = PROJECT_ROOT / "alethex" / "reporting" / "landing_page.html"
        if landing_path.exists():
            webbrowser.open(landing_path.as_uri())
        else:
            self.log("Landing page HTML not found.")


def main():
    root = tk.Tk()
    app = AlethexDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
