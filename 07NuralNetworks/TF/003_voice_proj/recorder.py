# recorder.py
# Console voice-identity recorder & manager per specified pseudocode.
# Usage: python recorder.py
# Deps:  pip install sounddevice soundfile numpy
# Audio: WAV @ 16kHz, mono, PCM_16

import sys
import os
import re
import threading
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


# -----------------------------
# Config
# -----------------------------
SR = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"

ROOT_DIR = Path("./data/voice_identities").resolve()
TMP_DIR = ROOT_DIR / "tmp"
ROOT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Cross-platform single-key read
# -----------------------------
def getch_blocking():
    """
    Wait for a single keypress (any key) and return the character (str).
    Windows: no Enter needed; POSIX: raw mode, no Enter needed.
    """
    try:
        # Windows
        import msvcrt
        ch = msvcrt.getch()
        try:
            ch = ch.decode("utf-8", errors="ignore")
        except Exception:
            ch = str(ch)
        return ch
    except ImportError:
        # macOS / Linux
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


def keypress_to_event(event):
    getch_blocking()
    event.set()


# -----------------------------
# Helpers: files & audio
# -----------------------------
def is_snake_case(name: str) -> bool:
    """
    snake_case: lowercase letters/digits separated by single underscores.
    Examples: abc, abc_123, a1_b2_c3
    """
    return re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", name) is not None


def create_empty_wav(path: Path):
    """Create (or truncate to) an empty WAV file with correct header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(path), mode="w", samplerate=SR, channels=CHANNELS, subtype=SUBTYPE):
        pass


def wav_is_blank(path: Path) -> bool:
    """True if file missing or has 0 frames (blank/header-only)."""
    if not path.exists():
        return True
    try:
        info = sf.info(str(path))
        return info.frames == 0
    except Exception:
        # If header-only or unreadable header: treat as blank
        try:
            return path.stat().st_size <= 44
        except Exception:
            return True


def wav_duration_seconds(path: Path) -> float:
    if wav_is_blank(path):
        return 0.0
    info = sf.info(str(path))
    if info.samplerate == 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def append_wav(dst_path: Path, src_path: Path):
    """Append src audio to dst and rewrite dst (16kHz, mono, PCM_16)."""
    # dst
    if not wav_is_blank(dst_path):
        dst_data, _ = sf.read(str(dst_path), dtype="int16", always_2d=True)
    else:
        dst_data = np.zeros((0, CHANNELS), dtype=np.int16)

    # src (must exist & not blank)
    if wav_is_blank(src_path):
        return
    src_data, _ = sf.read(str(src_path), dtype="int16", always_2d=True)

    combined = np.vstack((dst_data, src_data))
    with sf.SoundFile(str(dst_path), mode="w", samplerate=SR, channels=CHANNELS, subtype=SUBTYPE) as f:
        f.write(combined)


def erase_wav_contents(path: Path):
    """Truncate to an empty WAV file (keep header)."""
    create_empty_wav(path)


def play_wav(path: Path):
    """Play WAV file and block until finished."""
    if wav_is_blank(path):
        return
    data, fs = sf.read(str(path), dtype="float32", always_2d=False)
    sd.play(data, fs)
    sd.wait()


def trim_edges_wav(path: Path, seconds: float = 0.2):
    """Trim 'seconds' from start and end to avoid keypress sounds."""
    if wav_is_blank(path):
        return
    data, fs = sf.read(str(path), dtype="int16", always_2d=True)
    trim = int(max(0, round(seconds * fs)))
    if trim == 0 or data.shape[0] <= 2 * trim:
        # Too short to trim both ends; leave as-is
        return
    trimmed = data[trim:-trim]
    with sf.SoundFile(str(path), mode="w", samplerate=fs, channels=CHANNELS, subtype=SUBTYPE) as f:
        f.write(trimmed)


def record_to_file_interactive(out_path: Path) -> float:
    """
    Wait for any key to start; record until next key; save to out_path;
    then trim 0.2s from start/end.
    Returns duration (seconds) AFTER trimming.
    """
    print("Press any key to start recording ⏺ ...")
    _ = getch_blocking()

    print("Recording... ⏺")
    print("Press any key to stop!")

    stop_event = threading.Event()
    threading.Thread(target=keypress_to_event, args=(stop_event,), daemon=True).start()

    q = queue.Queue()
    frames_written = 0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        with sf.SoundFile(str(out_path), mode="w", samplerate=SR, channels=CHANNELS, subtype=SUBTYPE) as wav_file:
            with sd.InputStream(samplerate=SR, channels=CHANNELS, dtype="int16", callback=audio_callback):
                while not stop_event.is_set():
                    data = q.get()
                    wav_file.write(data)
                    frames_written += len(data)

                # drain queue
                while not q.empty():
                    data = q.get_nowait()
                    wav_file.write(data)
                    frames_written += len(data)
    except KeyboardInterrupt:
        print("\nInterrupted. Finalizing file…")

    # Trim 0.2s edges to avoid keypress sounds
    trim_edges_wav(out_path, seconds=0.2)

    duration = wav_duration_seconds(out_path)
    return duration


# -----------------------------
# UI helpers
# -----------------------------
def print_menu(voice_name: str):
    print("\nUser options: 'q'->quit, 'p'->play stored voice, 'l'->listen unsaved part, "
          "'s'->save recent part, 'd'->delete unsaved part, 'r'->record")
    print("Press a key: ", end="", flush=True)


def confirm_yn(prompt: str) -> bool:
    """Return True if user answers yes (y/yes)."""
    ans = input(f"{prompt} ").strip().lower()
    return ans in {"y", "yes"}


# -----------------------------
# Main flow
# -----------------------------
def main():
    print("Enter voice_identity_name (snake_case):")
    voice_name = input("> ").strip()

    # snake_case validation
    if not is_snake_case(voice_name):
        print('"voice_identity_name" must be in snake_case')
        sys.exit(1)

    main_wav = ROOT_DIR / f"{voice_name}.wav"
    tmp_wav = TMP_DIR / f"{voice_name}_tmp.wav"

    # If main file does not exist: create blank; reset tmp
    if not main_wav.exists():
        create_empty_wav(main_wav)
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except Exception:
                pass
        create_empty_wav(tmp_wav)
        print(f"creating voice identity - {voice_name}")

    print(f"System is ready...\nRecord and update your voice identity - {voice_name}, "
          f"following the key press option prompts!")

    while True:
        print_menu(voice_name)
        key = getch_blocking()
        if not key:
            continue
        k = key.lower()
        print(k)  # echo the pressed key

        # Switch/case on k
        if k == 'q':
            if confirm_yn("Exit program? Press Y/N..."):
                print("Exit program as per user selection...")
                sys.exit(0)
            # else continue loop
        elif k == 'p':
            if wav_is_blank(main_wav):
                print(f"Voice identity {voice_name} is blank as of now! Record your voice identity first...")
            else:
                dur = wav_duration_seconds(main_wav)
                print(f'Playing voice identity {voice_name}: total_time of {voice_name}.wav {dur:.2f} seconds, '
                      f'16KHz, "Mono", "{SUBTYPE}"')
                play_wav(main_wav)
        elif k == 'l':
            if wav_is_blank(tmp_wav):
                print(f"Nothing new is recorded yet for voice identity {voice_name}! Record your voice first...")
            else:
                dur = wav_duration_seconds(tmp_wav)
                print(f'Playing recent recording for voice identity {voice_name}: '
                      f'total_time_of {voice_name}_tmp.wav {dur:.2f} seconds, 16KHz, "Mono", "{SUBTYPE}"')
                play_wav(tmp_wav)
        elif k == 's':
            if wav_is_blank(tmp_wav):
                print(f"Nothing new is recorded yet for voice identity {voice_name}! Record your voice first...")
            else:
                if confirm_yn(f"Save recent recording to voice identity {voice_name}? Press Y/N..."):
                    print(f"Saving recent recording to voice identity {voice_name}...")
                    append_wav(main_wav, tmp_wav)
                    erase_wav_contents(tmp_wav)
                    print(f"voice_identity {voice_name} updated successfully!")
                else:
                    print("Not saving recent recording as per user selection.\n"
                          "Not deleting either. To delete, specifically select 'd'/'D' option...")
        elif k == 'd':
            if wav_is_blank(tmp_wav):
                print(f"Nothing new is recorded yet for voice identity {voice_name}! Nothing to delete...")
            else:
                if confirm_yn(f"Delete recent recording for voice identity {voice_name}? Press Y/N..."):
                    erase_wav_contents(tmp_wav)
                    print(f"Deleted recent recording to voice identity {voice_name} as per user selection...")
                else:
                    print("Not deleting recent recording as user selection was not 'Y' or 'y'...")
        elif k == 'r':
            # If tmp not blank, ask to overwrite
            if not wav_is_blank(tmp_wav):
                if not confirm_yn(f'"{tmp_wav}" is not blank! Overwrite the same? Y/N?'):
                    print(f"Not changing voice identity {voice_name} as per user selection...")
                    # back to menu
                    continue
                erase_wav_contents(tmp_wav)

            # Record interactive
            print("Press any key to start recording ⏺ ...")
            duration = record_to_file_interactive(tmp_wav)
            print(f"Recording successful! total_time_of_track in seconds ({duration:.2f}), "
                  f"SamplingRate=16KHz, Mono, {SUBTYPE}")
        else:
            # unknown key -> re-loop
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting on user interrupt. Goodbye!")
