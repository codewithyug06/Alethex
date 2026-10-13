"""Tests for alethex.banner — rendering, suppression, and env-var gates."""

import io
import os
import sys

import pytest

from alethex.banner import render_banner, should_show, print_banner


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakePipe(io.StringIO):
    def isatty(self) -> bool:
        return False


# ── render_banner ─────────────────────────────────────────────────────────────

def test_render_returns_string():
    assert isinstance(render_banner(width=80), str)


def test_render_contains_project_name():
    result = render_banner(width=80)
    assert "ALETHEX" in result or "alethex" in result.lower()


def test_render_narrow_terminal():
    result = render_banner(width=40)
    assert isinstance(result, str) and len(result) > 0


def test_render_very_narrow_terminal():
    result = render_banner(width=20)
    assert isinstance(result, str)


def test_render_wide_terminal():
    result = render_banner(width=160)
    assert isinstance(result, str)


def test_render_no_ansi_when_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b[" not in render_banner(width=80)


def test_render_no_ansi_when_dumb_term(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert "\x1b[" not in render_banner(width=80)


# ── should_show ───────────────────────────────────────────────────────────────

def test_should_show_false_flag():
    assert should_show(no_banner=True) is False


def test_should_show_false_env(monkeypatch):
    monkeypatch.setenv("ALETHEX_NO_BANNER", "1")
    assert should_show() is False


def test_should_show_false_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert should_show() is False


def test_should_show_false_pipe(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakePipe())
    assert should_show() is False


# ── print_banner ──────────────────────────────────────────────────────────────

def test_print_banner_silent_env(monkeypatch, capsys):
    monkeypatch.setenv("ALETHEX_NO_BANNER", "1")
    print_banner()
    assert capsys.readouterr().out == ""


def test_print_banner_silent_pipe(monkeypatch):
    buf = _FakePipe()
    monkeypatch.setattr(sys, "stdout", buf)
    print_banner()
    assert buf.getvalue() == ""


def test_print_banner_writes_to_tty(monkeypatch):
    buf = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.delenv("ALETHEX_NO_BANNER", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    print_banner()
    assert len(buf.getvalue()) > 0


def test_print_banner_no_banner_flag(monkeypatch):
    buf = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", buf)
    print_banner(no_banner=True)
    assert buf.getvalue() == ""
