import sys
import subprocess
import torch
import os
import warnings

warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        module="huggingface_hub")

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
        print("✅ MeCab imported successfully.")
    except ImportError as e:
        print("❌ MeCab is not installed.")
        print("Exception details:", e, "")

    # --- UniDic ---
    try:
        import MeCab
        # Attempt to initialize a tagger with UniDic
        tagger = MeCab.Tagger(".venv/Lib/site-packages/unidic")
        print("📦 UniDic dictionary available for MeCab.")
    except RuntimeError as e:
        print("❌ UniDic dictionary not found for MeCab.")
        print("Exception details:", e)
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
        print("📦 'averaged_perceptron_tagger_eng' found.")
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

    # --- Sound I/O: sounddevice + soundfile ---
    try:
        import sounddevice as sd
        import soundfile as sf
        print("✅ sounddevice imported successfully.")
        print("✅ soundfile imported successfully.")

        # Versions / backend
        try:
            pa_ver = sd.get_portaudio_version()
            pa_ver_str = pa_ver[1] if isinstance(pa_ver, tuple) and len(pa_ver) > 1 else str(pa_ver)
            print(f"🎚️ PortAudio backend: {pa_ver_str}")
        except Exception as e:
            print("⚠️ Could not query PortAudio version:", e)

        # Default devices
        try:
            default_in, default_out = (sd.default.device or (None, None))
            print(f"🎤 Default input device index: {default_in}, output index: {default_out}")
            if default_in is not None and default_in >= 0:
                dev = sd.query_devices(default_in)
                host = sd.query_hostapis(dev['hostapi'])['name']
                print(f"   Input device: {dev['name']} (host API: {host})")
        except Exception as e:
            print("⚠️ Could not query default audio devices:", e)

        # List a few input devices
        try:
            devices = sd.query_devices()
            input_devs = [
                f"[{i}] {d['name']} ({sd.query_hostapis(d['hostapi'])['name']})"
                for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0
            ]
            print(f"🎛️ Input devices found: {len(input_devs)}")
            for line in input_devs[:5]:
                print("   •", line)
            if len(input_devs) > 5:
                print("   … (showing first 5)")
        except Exception as e:
            print("⚠️ Could not enumerate audio devices:", e)

        # Check microphone supports 16 kHz / 22.05 kHz mono
        for sr in (16000, 22050):
            try:
                sd.check_input_settings(samplerate=sr, channels=1)
                print(f"✅ Mic supports mono {sr} Hz input.")
            except Exception as e:
                print(f"❌ Mic may not support mono {sr} Hz input:", e)

        # Check soundfile can write WAV/PCM_16
        try:
            wav_ok = "WAV" in sf.available_formats()
            pcm16_ok = "PCM_16" in sf.available_subtypes("WAV")
            if wav_ok and pcm16_ok:
                print("✅ soundfile supports WAV / PCM_16 writing.")
            else:
                print("❌ soundfile missing WAV/PCM_16 support.")
        except Exception as e:
            print("⚠️ Could not query soundfile formats:", e)

        # Optional quick mic test (2s @ 16k mono) — enable with DIAG_MIC_TEST=1
        if os.getenv("DIAG_MIC_TEST") == "1":
            out = "./data/diag_mic_test.wav"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            duration = float(os.getenv("DIAG_MIC_TEST_SECONDS", "2.0"))
            print(f"🎙️ Recording {duration}s at 16k mono to {out} ...")
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
            sf.write(out, audio, 16000, subtype="PCM_16")
            print("✅ Mic test recorded:", out)

    except ImportError as e:
        print("❌ ERROR: Could not import sounddevice and/or soundfile.")
        print("Exception details:", e)
        print("   Try: pip install sounddevice soundfile")

    # --- Model checkpoints ---
    ckpt_path = os.path.join(os.getcwd(), "OpenVoice", "checkpoints")
    if os.path.exists(ckpt_path):
        ckpts = os.listdir(ckpt_path)
        print("📂 Checkpoints found:", ckpts if ckpts else "None found")
    else:
        print("❌ No 'checkpoints/' folder found.")

if __name__ == "__main__":
    check()
