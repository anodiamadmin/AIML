# pip install sounddevice soundfile
import sounddevice as sd, soundfile as sf

SR = 16000        # or 22050
SECONDS = 90      # change to desired duration
CHANNELS = 1

print("Recording...")
audio = sd.rec(int(SECONDS*SR), samplerate=SR, channels=CHANNELS, dtype='int16')
sd.wait()
sf.write("./data/user_voice.wav", audio, SR, subtype="PCM_16")
print("Saved ./data/user_voice.wav")
