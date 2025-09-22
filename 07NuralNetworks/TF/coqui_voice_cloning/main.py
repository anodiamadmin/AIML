from TTS.api import TTS

# load a multilingual voice cloning model (XTTS v2)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# read text from file
with open("./data/script.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

# generate speech in your voice
tts.tts_to_file(
    text=text_content,
    file_path="./data/speech.wav",
    speaker_wav="./data/voice_identities/anirban_id.wav",  # your reference audio
    language="en",
    split_sentences=True
)
