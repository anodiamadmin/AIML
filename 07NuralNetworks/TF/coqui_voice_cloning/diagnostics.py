#!/usr/bin/env python3
"""
diagnostic.py — sanity checks for Coqui TTS + PyTorch (CPU only)
"""

import sys, shutil, subprocess, importlib
from pathlib import Path

DATA_DIR = Path("./data")
VOICE_FILE = DATA_DIR/ "voice_identities" / "anirban_id.wav"
SCRIPT_FILE = DATA_DIR / "script.txt"

def ok(msg): print(f"[  OK  ] {msg}")
def fail(msg): print(f"[FAIL] {msg}")

def check_python():
    print("== Python ==")
    print(sys.version)
    if sys.version_info.minor in (8,9,10,11):
        ok("Python version supported (3.8–3.11).")
    else:
        fail("Python version may be incompatible with Coqui TTS.")

def check_package(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
        ok(f"{pkg} installed")
        return True
    except Exception as e:
        fail(f"{pkg} missing or broken: {e}")
        return False

def check_torch():
    try:
        import torch
        ok(f"torch {torch.__version__} imported")
        if torch.cuda.is_available():
            fail("CUDA detected — you expected CPU only, check env!")
        else:
            ok("CPU-only mode confirmed")
    except Exception as e:
        fail(f"torch import failed: {e}")

def check_ffmpeg():
    if shutil.which("ffmpeg"):
        try:
            out = subprocess.check_output(["ffmpeg", "-version"], text=True)
            ok("ffmpeg available: " + out.splitlines()[0])
        except Exception:
            ok("ffmpeg found")
    else:
        fail("ffmpeg not in PATH")

def check_tts():
    try:
        from TTS.api import TTS
        ok("Coqui TTS API imported")
        tts = TTS()
        models = tts.list_models()
        ok(f"TTS.list_models() returned {len(models)} models")
    except Exception as e:
        fail(f"TTS test failed: {e}")

def check_data():
    if VOICE_FILE.exists():
        ok(f"Voice file present: {VOICE_FILE}")
    else:
        fail(f"Missing voice file: {VOICE_FILE}")
    if SCRIPT_FILE.exists():
        ok(f"Transcript present: {SCRIPT_FILE}")
    else:
        fail(f"Missing transcript: {SCRIPT_FILE}")

def main():
    check_python()
    check_package("soundfile")
    check_package("librosa")
    check_package("numpy")
    check_torch()
    check_ffmpeg()
    check_tts()
    check_data()
    print("\nDiagnostics complete.")

if __name__ == "__main__":
    main()
