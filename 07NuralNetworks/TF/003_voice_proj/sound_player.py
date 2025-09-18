# sound_player.py
# Usage:
#   python sound_player.py "./any/relative/path/sound.mp3"
#
# Deps:
#   pip install pydub sounddevice
#   For MP3 decoding: install FFmpeg and ensure 'ffmpeg' is on PATH.

import argparse
import sys
import shutil
from pathlib import Path

import numpy as np
from pydub import AudioSegment
import sounddevice as sd


def _dtype_from_sample_width(width_bytes: int):
    # pydub typically produces 2-byte (16-bit) samples; handle common cases.
    return {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}.get(width_bytes, np.int16)


def play_audio(file_path: Path) -> None:
    # Load with pydub (uses ffmpeg for mp3)
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"❌ Could not load '{file_path}': {e}")
        if file_path.suffix.lower() == ".mp3" and shutil.which("ffmpeg") is None:
            print("👉 MP3 playback requires FFmpeg. Install it and ensure 'ffmpeg' is on your PATH.")
        sys.exit(1)

    duration_sec = len(audio) / 1000.0  # pydub length is in ms
    print(f"▶ Playing: {file_path}")
    print(f"   Specs: {duration_sec:.2f}s, {audio.frame_rate} Hz, {audio.channels} ch, "
          f"{8 * audio.sample_width}-bit")
    print("   (Press Ctrl+C to stop)")

    # Convert to numpy and play via sounddevice
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    samples = samples.astype(_dtype_from_sample_width(audio.sample_width), copy=False)

    try:
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\n⏹ Stopped by user.")
    except Exception as e:
        print(f"❌ Playback error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Play a WAV or MP3 file from the console.")
    parser.add_argument("path", help="Path to .wav or .mp3 file")
    args = parser.parse_args()

    p = Path(args.path).expanduser()
    # resolve only after confirming existence (keeps nicer error if path is wrong)
    if not p.exists():
        print(f"❌ File not found: {p}")
        sys.exit(1)
    p = p.resolve()

    if p.suffix.lower() not in {".wav", ".mp3"}:
        print(f"⚠️ '{p.suffix}' not explicitly supported; attempting to play anyway…")

    if p.suffix.lower() == ".mp3" and shutil.which("ffmpeg") is None:
        print("⚠️ FFmpeg not detected. MP3 decoding may fail. "
              "Install FFmpeg and ensure 'ffmpeg' is on your PATH if this doesn't play.")

    play_audio(p)


if __name__ == "__main__":
    main()
