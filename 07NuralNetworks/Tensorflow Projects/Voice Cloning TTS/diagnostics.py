import sys
import subprocess
import torch
import os

def check():
    print("🔎 Python version:", sys.version)

    try:
        import openvoice
        print("✅ OpenVoice imported successfully.")
    except ImportError:
        print("❌ ERROR: Could not import OpenVoice.")

    # GPU/CPU check
    print("💻 Torch version:", torch.__version__)
    print("⚡ CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("   (Running on CPU mode)")

    # FFmpeg
    try:
        completed = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print("🎵 FFmpeg installed:", completed.returncode == 0)
    except FileNotFoundError:
        print("❌ ERROR: FFmpeg not found.")

    # Check model checkpoints
    ckpt_path = os.path.join(os.getcwd(), "OpenVoice/checkpoints")
    if os.path.exists(ckpt_path):
        ckpts = os.listdir(ckpt_path)
        print("📂 Checkpoints found:", ckpts if ckpts else "None found")
    else:
        print("❌ No 'checkpoints/' folder found.")

if __name__ == "__main__":
    check()
