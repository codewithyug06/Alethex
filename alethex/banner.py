"""
ALETHEX terminal banner.

Displayed once at CLI launch. Never on import. Never in CI/pipes.
Suppress with:  ALETHEX_NO_BANNER=1  or  alethex --no-banner <cmd>
"""

import os
import shutil
import sys
from typing import Tuple


# ── colour capability ─────────────────────────────────────────────────────────

def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _color_ok() -> bool:
    if not _is_tty():
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    return True


def _truecolor_ok() -> bool:
    return os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def _reset() -> str:
    return "\x1b[0m"


def _bold_on() -> str:
    return "\x1b[1m"


def _rgb_fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def _ansi16_fg(code: int) -> str:
    return f"\x1b[{code}m"


# ── gradient math ─────────────────────────────────────────────────────────────

_CYAN   = (0,   240, 240)
_BLUE   = (0,   100, 255)
_VIOLET = (148,   0, 255)


def _lerp3(
    t: float,
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    c3: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """3-stop linear gradient; t in [0, 1]: c1 → c2 → c3."""
    if t <= 0.5:
        s, a, b = t * 2.0, c1, c2
    else:
        s, a, b = (t - 0.5) * 2.0, c2, c3
    return (
        int(a[0] + s * (b[0] - a[0])),
        int(a[1] + s * (b[1] - a[1])),
        int(a[2] + s * (b[2] - a[2])),
    )


def _paint(text: str, t: float, truecolor: bool, bold: bool = False) -> str:
    r, g, b = _lerp3(t, _CYAN, _BLUE, _VIOLET)
    prefix = (_bold_on() if bold else "") + (
        _rgb_fg(r, g, b) if truecolor else _ansi16_fg(96 if t < 0.35 else 94 if t < 0.65 else 95)
    )
    return f"{prefix}{text}{_reset()}"


# ── ASCII artwork ─────────────────────────────────────────────────────────────

# 13-row block-art "A" emblem (≈38 chars wide)
_LOGO = [
    "               ▄░▄               ",
    "             ▄▓███▓▄             ",
    "            ▄█▓▀ ▀▓█▄            ",
    "           ▄█▓▀   ▀▓█▄           ",
    "          ▄█▓▀     ▀▓█▄          ",
    "         ▄█▓▀       ▀▓█▄         ",
    "        ▄█▓▀         ▀▓█▄        ",
    "       ▄█▓█▓▓▓▓▓▓▓▓▓▓▓█▓█▄      ",
    "      ▄█▓▀               ▀▓█▄   ",
    "     ▄█▓▀                 ▀▓█▄  ",
    "    ▄███▄                 ▄███▄ ",
    "   ▄█▓▀▀█▄               ▄█▀▀▓█▄",
    "  ▄█████████████████████████████▄",
]

# 5-row block-font "ALETHEX"
_WORDMARK = [
    "  ▄█▄   █      ████  ██████  █  █   ████  ▄   ▄ ",
    " █   █  █      █       █     █  █   █     ██  █  ",
    " █████  █      ████    █     ████   ████  ████   ",
    " █   █  █      █       █     █  █   █     █ ██   ",
    " █   █  █████  ████    █     █  █   ████  █   █  ",
]

_TAGLINE  = "T R U T H   G U A R D S   I N T E L L I G E N C E"
_SUBLINE  = "Universal AI Memory & Truth Guard"
_FEATURES = "NLI Engine  ·  Temporal Memory  ·  Consistency Analysis"


# ── render ────────────────────────────────────────────────────────────────────

def _ctr(text: str, w: int) -> str:
    pad = max(0, (w - len(text)) // 2)
    return " " * pad + text


def render_banner(width: int = 0) -> str:
    """Build and return the banner string. width=0 → auto-detect."""
    tw = width or shutil.get_terminal_size((80, 24)).columns
    color = _color_ok()
    tc    = _truecolor_ok() if color else False
    out: list[str] = []

    # ── emblem ────────────────────────────────────────────────────────────────
    if tw >= 50:
        n = len(_LOGO)
        for i, row in enumerate(_LOGO):
            t = i / max(n - 1, 1)
            line = _ctr(row, tw)
            out.append(_paint(line, t, tc) if color else line)
        out.append("")
    else:
        # narrow: skip large art
        title = "[ ALETHEX ]"
        line = _ctr(title, tw)
        out.append(_paint(line, 0.4, tc, bold=True) if color else line)
        out.append("")

    # ── wordmark ──────────────────────────────────────────────────────────────
    if tw >= 56:
        nw = len(_WORDMARK)
        for i, row in enumerate(_WORDMARK):
            t = 0.42 + 0.3 * (i / max(nw - 1, 1))
            line = _ctr(row, tw)
            out.append(_paint(line, t, tc, bold=True) if color else line)
        out.append("")

    # ── tagline ───────────────────────────────────────────────────────────────
    tl = _ctr(_TAGLINE, tw)
    out.append(_paint(tl, 0.45, tc, bold=True) if color else tl)
    out.append("")

    # ── separator ─────────────────────────────────────────────────────────────
    sep = "─" * min(tw, 60)
    out.append(_ctr(_paint(sep, 0.5, tc) if color else sep, tw))
    out.append("")

    # ── info ──────────────────────────────────────────────────────────────────
    for txt in (_SUBLINE, _FEATURES):
        line = _ctr(txt, tw)
        out.append(_paint(line, 0.3, tc) if color else line)
    out.append("")

    # ── version ───────────────────────────────────────────────────────────────
    try:
        from importlib.metadata import version as _v
        ver = _v("alethex")
    except Exception:
        try:
            from alethex import __version__ as ver  # type: ignore[attr-defined]
        except Exception:
            ver = "dev"

    ready = f"alethex v{ver}  ·  System ready."
    rl = _ctr(ready, tw)
    out.append(_paint(rl, 0.55, tc) if color else rl)
    out.append("")

    return "\n".join(out)


# ── public API ────────────────────────────────────────────────────────────────

def should_show(no_banner: bool = False) -> bool:
    """Return True when the banner should be printed."""
    if no_banner:
        return False
    if os.environ.get("ALETHEX_NO_BANNER"):
        return False
    if os.environ.get("CI"):
        return False
    if not _is_tty():
        return False
    return True


def print_banner(no_banner: bool = False) -> None:
    """Print the banner to stdout when conditions allow."""
    if not should_show(no_banner):
        return
    sys.stdout.write(render_banner())
    sys.stdout.flush()
