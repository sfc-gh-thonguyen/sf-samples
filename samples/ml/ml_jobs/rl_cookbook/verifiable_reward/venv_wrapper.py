#!/usr/bin/env python3
"""Wrapper to exec into the AReaL venv Python.

ML Jobs run entrypoints with the Container Runtime's system Python.
Since AReaL is installed in /AReaL/.venv/, this wrapper re-execs
the given script under the venv Python so all AReaL imports work.

Usage: python venv_wrapper.py <script.py> [args...]
"""
import os
import sys

VENV_PYTHON = "/AReaL/.venv/bin/python3"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python venv_wrapper.py <script.py> [args...]")
        sys.exit(1)

    script = sys.argv[1]
    args = sys.argv[2:]

    if os.path.exists(VENV_PYTHON):
        print(f"[venv_wrapper] exec {VENV_PYTHON} -u {script} {' '.join(args)}")
        os.execv(VENV_PYTHON, [VENV_PYTHON, "-u", script] + args)
    else:
        # Fallback: try running with current Python (maybe AReaL is in system path)
        print(f"[venv_wrapper] WARN: {VENV_PYTHON} not found, using {sys.executable}")
        os.execv(sys.executable, [sys.executable, "-u", script] + args)
