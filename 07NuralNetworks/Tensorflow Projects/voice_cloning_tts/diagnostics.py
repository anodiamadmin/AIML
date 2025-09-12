import sys
import subprocess
import torch
import os
import warnings

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="librosa") # Suppresses deprecation warning for "pkg_resources" from "setuptools"


print("🚀 Running diagnostics script...")
print("📂 diagnostics.py is located at:", __file__)

def check():
    # --- Python version ---
    print("🔎 Python version:", sys.version)

    # --- OpenVoice ---
    try:
        import OpenVoice.openvoice
        print("✅ openvoice imported successfully.")
    except ImportError:
        print("❌ ERROR: Could not import openvoice.")

    # --- MeloTTS ---
    try:
        import melo
        from melo.api import TTS
        print("✅ MeloTTS imported successfully from melo.api\n✅ MeloTTS base module imported successfully.")
    except ImportError:
        print("❌ ERROR: Could not import MeloTTS base module.\n❌ ERROR: Could not import MeloTTS from melo.api.")

    # --- MeCab ---
    try:
        import MeCab
        print("✅ MeCab imported successfully.\n")
    except ImportError as e:
        print("❌ MeCab is not installed.")
        print("Exception details:", e, "\n")

    # --- UniDic ---
    try:
        import MeCab
        # Attempt to initialize a tagger with UniDic
        tagger = MeCab.Tagger(".venv/Lib/site-packages/unidic")
        print("📦 UniDic dictionary available for MeCab.\n")
    except RuntimeError as e:
        print("❌ UniDic dictionary not found for MeCab.")
        print("Exception details:", e, "\n")
    except ImportError:
        # MeCab not installed, skip this check
        pass

    # --- NLTK ---
    # 1. Import check
    try:
        import nltk
        print("✅ NLTK imported successfully.")
    except ImportError as e:
        print("❌ ERROR: Could not import NLTK.")
        print("Exception details:", e)

    # 2. 'punkt' tokenizer check
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
        print("📦 NLTK 'punkt' tokenizer available.")
    except LookupError as e:
        print("❌ ERROR: NLTK 'punkt' tokenizer not found.")
        print("Run: nltk.download('punkt')")
        print("Exception details:", e)

    # 3. 'averaged_perceptron_tagger_eng' check
    try:
        import nltk
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        print("📦 'averaged_perceptron_tagger_eng' found.\n")
    except LookupError as e:
        print("❌ ERROR: 'averaged_perceptron_tagger_eng' not found.")
        print("Run: nltk.download('averaged_perceptron_tagger_eng')")
        print("Exception details:", e)

    # --- Torch / CUDA ---
    print("💻 Torch version:", torch.__version__)
    print("⚡ CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("   (Running on CPU mode)")

    # --- FFmpeg ---
    try:
        completed = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print("🎵 FFmpeg installed:", completed.returncode == 0)
    except FileNotFoundError:
        print("❌ ERROR: FFmpeg not found.")

    # --- Model checkpoints ---
    ckpt_path = os.path.join(os.getcwd(), "OpenVoice", "checkpoints")
    if os.path.exists(ckpt_path):
        ckpts = os.listdir(ckpt_path)
        print("📂 Checkpoints found:", ckpts if ckpts else "None found")
    else:
        print("❌ No 'checkpoints/' folder found.")

if __name__ == "__main__":
    check()
