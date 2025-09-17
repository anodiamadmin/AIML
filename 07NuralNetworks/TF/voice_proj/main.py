import torch
import nltk
from pathlib import Path
import shutil

# ------------------------
# NLTK setup
# ------------------------

# Create venv-specific nltk_data folder
NLTK_DATA_DIR = Path(".venv") / "nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True)

# Tell NLTK where to look
nltk.data.path.append(str(NLTK_DATA_DIR))

# Download required resources if missing
for resource in ["averaged_perceptron_tagger", "punkt"]:
    try:
        nltk.data.find(f"taggers/{resource}" if "tagger" in resource else f"tokenizers/{resource}")
    except LookupError:
        print(f"Downloading NLTK resource: {resource} ...")
        nltk.download(resource, download_dir=str(NLTK_DATA_DIR))

# 🔧 Fix g2p_en expecting "averaged_perceptron_tagger_eng"
tagger_src = NLTK_DATA_DIR / "taggers" / "averaged_perceptron_tagger"
tagger_dst = NLTK_DATA_DIR / "taggers" / "averaged_perceptron_tagger_eng"

if tagger_src.exists() and not tagger_dst.exists():
    print("🔧 Creating copy for averaged_perceptron_tagger_eng...")
    shutil.copytree(tagger_src, tagger_dst)

# ------------------------
# OpenVoice / MeloTTS imports
# ------------------------

from OpenVoice.openvoice import se_extractor
from OpenVoice.openvoice.api import ToneColorConverter
from melo.api import TTS

# ------------------------
# Paths
# ------------------------
DATA_DIR = Path("./data")
VOICE_FILE = DATA_DIR / "my_voice.wav"
SCRIPT_FILE = DATA_DIR / "script.txt"
OUTPUT_FILE = DATA_DIR / "speech.mp3"

CHECKPOINTS_DIR = Path("./OpenVoice/checkpoints")
CONVERTER_CKPT = CHECKPOINTS_DIR / "converter"
BASE_SPEAKERS = CHECKPOINTS_DIR / "base_speakers"

TMP_AUDIO = DATA_DIR / "tmp.wav"  # temporary TTS output

# ------------------------
# Sanity checks
# ------------------------
for path in [VOICE_FILE, SCRIPT_FILE, CONVERTER_CKPT, BASE_SPEAKERS]:
    if not path.exists():
        raise FileNotFoundError(f"Required file or folder not found: {path}")

# ------------------------
# Load transcript
# ------------------------
with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
    transcript_text = f.read().strip()
if not transcript_text:
    raise ValueError("Transcript file is empty!")

# ------------------------
# Device
# ------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------
# Initialize ToneColorConverter
# ------------------------
tone_color_converter = ToneColorConverter(
    f"{CONVERTER_CKPT}/config.json",
    device=device
)
tone_color_converter.load_ckpt(f"{CONVERTER_CKPT}/checkpoint.pth")

# ------------------------
# Extract speaker embedding from your voice
# ------------------------
print("🎤 Extracting speaker embedding from reference voice...")
target_se, _ = se_extractor.get_se(str(VOICE_FILE), tone_color_converter, vad=True)
print("✅ Speaker embedding extracted.")

# ------------------------
# English-only TTS
# ------------------------
print("📝 Synthesizing text using English MeloTTS...")

# Force English language
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id

# Use first available speaker in English base model
speaker_id = next(iter(speaker_ids.values()))

# Generate temporary WAV
model.tts_to_file(transcript_text, speaker_id, str(TMP_AUDIO))
print(f"✅ Temporary TTS output saved to {TMP_AUDIO}")

# ------------------------
# Apply cloned voice using ToneColorConverter
# ------------------------
print("🔊 Converting to cloned voice...")
tone_color_converter.convert(
    audio_src_path=str(TMP_AUDIO),
    src_se=torch.load(f"{BASE_SPEAKERS}/ses/en-newest.pth", map_location=device),
    tgt_se=target_se,
    output_path=str(OUTPUT_FILE),
    message="@MyShell"
)
print(f"✅ Cloned speech saved to: {OUTPUT_FILE}")

# Remove temporary WAV
# TMP_AUDIO.unlink(missing_ok=True)
