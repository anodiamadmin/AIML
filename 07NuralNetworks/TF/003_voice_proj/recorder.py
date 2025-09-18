# recorder.py
# pip install sounddevice soundfile numpy noisereduce
import sys, os, threading, queue
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects


# -----------------------------
# Audio & Paths Config
# -----------------------------
SR = 16000           # or 22050
CHANNELS = 1
SUBTYPE = "PCM_16"   # 16-bit PCM
ROOT_DIR = Path("./data/voice_identities")
ROOT_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_VOICE_NAME = None

# -----------------------------
# Cleanup if user quits
# -----------------------------
def clean_before_quit(voice_name: str):
    """
    Best-effort cleanup for this session's files.
    - Always delete <voice_name>_tmp.wav if present.
    - Delete <voice_name>.wav ONLY if it's empty (0 frames) or header-only.
    Safe no-op if voice_name is None/blank.
    """
    try:
        if not voice_name:
            return

        main_path = ROOT_DIR / f"{voice_name}.wav"
        tmp_path  = ROOT_DIR / f"{voice_name}_tmp.wav"

        # Remove temp take (always)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
                print(f"🧹 Removed temp: {tmp_path.name}")
            except Exception as e:
                print(f"⚠️ Could not remove temp '{tmp_path.name}': {e}")

        # Remove main only if it's empty / header-only
        if main_path.exists():
            remove_main = False
            try:
                info = sf.info(str(main_path))
                if info.frames == 0:
                    remove_main = True
            except Exception:
                # Fallback: tiny header-only files are ~44 bytes
                try:
                    if main_path.stat().st_size <= 44:
                        remove_main = True
                except Exception:
                    pass

            if remove_main:
                try:
                    main_path.unlink()
                    print(f"🧹 Removed empty main: {main_path.name}")
                except Exception as e:
                    print(f"⚠️ Could not remove main '{main_path.name}': {e}")

    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")


# -----------------------------
# Cross-platform single-key read
# -----------------------------
def getch_blocking():
    """
    Wait for a single keypress (any key), cross-platform, and return the character (str).
    On Windows, no Enter is needed; on POSIX we put the terminal into raw mode.
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
# File helpers
# -----------------------------
def create_empty_wav(path: Path):
    """
    Create an empty WAV file with the given audio format (header only, no samples).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(path), mode="w", samplerate=SR, channels=CHANNELS, subtype=SUBTYPE) as _:
        pass

def wav_duration_seconds(path: Path) -> float:
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    info = sf.info(str(path))
    if info.samplerate == 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)

def append_wav(dst_path: Path, src_path: Path):
    """
    Append src WAV audio to dst WAV audio and rewrite dst with combined data.
    Keeps format SR / CHANNELS / SUBTYPE.
    """
    # Read existing destination (if any)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        dst_data, _ = sf.read(str(dst_path), dtype="int16", always_2d=True)
    else:
        dst_data = np.zeros((0, CHANNELS), dtype=np.int16)

    # Read source
    if not (src_path.exists() and src_path.stat().st_size > 0):
        return
    src_data, _ = sf.read(str(src_path), dtype="int16", always_2d=True)

    combined = np.vstack((dst_data, src_data))
    with sf.SoundFile(str(dst_path), mode="w", samplerate=SR, channels=CHANNELS, subtype=SUBTYPE) as f:
        f.write(combined)

# -----------------------------
# Recording & Playback
# -----------------------------
def record_to_file(out_path: Path) -> float:
    """
    Records microphone audio to out_path until any key is pressed.
    Returns duration (seconds).
    """
    print("\n▶ Recording ⏺ in progress ▶")
    print("Press any key to stop ⏹")

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

    duration = frames_written / float(SR)
    return duration

def play_wav(path: Path):
    """
    Plays the WAV file and blocks until playback finishes.
    """
    if not path.exists() or path.stat().st_size == 0:
        print("⚠️ Nothing to play.")
        return

    data, fs = sf.read(str(path), dtype="float32", always_2d=False)
    sd.play(data, fs)
    sd.wait()

# -----------------------------
# UI helpers
# -----------------------------
def print_banner():
    print("Create your unique voice identity.")
    print("Press 'q+↵' to quit at any point.")
    print("Enter a name for your voice_identity using snake_case_format:")

def ask_for_voice_identity_name() -> str:
    while True:
        name = input("> ").strip()

        # allow quit
        if name.lower() == "q":
            clean_before_quit('') # nothing to clean yet, but safe no-op
            print("👋 Exiting. No voice identity was created.")
            sys.exit(0)

        if not name:
            print("⚠️ voice_identity name cannot be blank.")
            continue

        # Ensure snake_case-ish (soft check; you can harden if you want)
        # Here we just strip spaces; user asked for snake_case format.
        candidate = f"{name}.wav" if not name.endswith(".wav") else name
        base = candidate[:-4] if candidate.endswith(".wav") else candidate
        if " " in base:
            print("⚠️ Please use snake_case (no spaces). Try again:")
            continue

        # Existence check in ./data/voice_identities
        main_path = ROOT_DIR / (base + ".wav" if not candidate.endswith(".wav") else candidate)
        if main_path.exists():
            print(f"⚠️ '{main_path.name}' already exists in '{ROOT_DIR}'. Please enter another name:")
            continue

        return base if not base.endswith(".wav") else base[:-4]

def prompt_press_any_key_or_quit_line(voice_name: str):
    """
    Show the 'Press any key to start recording' message and let user quit with 'q'.
    We accept a single 'q' (no Enter needed) for immediate exit here.
    """
    print("\n▶ Press any key to start recording ⏺")
    ch = getch_blocking()
    if ch and ch.lower() == "q":
        clean_before_quit(voice_name)
        print("👋 Exiting. Goodbye!")
        sys.exit(0)

def prompt_review_menu(tmp_path: Path, voice_name: str, duration: float):
    """
    After a take finishes recording, show the review menu and handle:
    - 'P' Play
    - 'D' Delete and re-record
    - 'A' Accept (append to main and delete tmp)
    - 'q' Quit
    Return one of: 'delete', 'accepted', 'repeat'
    """
    print(f"\n{voice_name} has been temporarily recorded.")
    print("Press 'P' to play ▶")
    print("'D' to delete and re-record ⏺")
    print("(or 'q' to quit)")

    while True:
        ch = getch_blocking()
        if not ch:
            continue
        c = ch.lower()

        if c == "q":
            clean_before_quit(voice_name)
            print("👋 Exiting as requested.")
            sys.exit(0)

        elif c == "d":
            # Delete temp and ask to re-record
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as e:
                    print(f"⚠️ Could not delete temp file: {e}")
            print("⏹ Last recording deleted ⏹")
            print("▶ Press any key to re-start recording ⏺")
            # Any key restarts (also allow 'q' to quit)
            ch2 = getch_blocking()
            if ch2 and ch2.lower() == "q":
                clean_before_quit(voice_name)
                print("👋 Exiting. Goodbye!")
                sys.exit(0)
            return "repeat"

        elif c == "p":
            # Play and show specs, then secondary menu
            dur = wav_duration_seconds(tmp_path)
            print(f"\n▶ Playing ( {dur:.2f}s, {SR} Hz, mono, {SUBTYPE} )")
            play_wav(tmp_path)

            # Post-play menu
            print("\nPress 'P' to play again ▶")
            print("'D' to delete and re-record ⏺")
            print("'A' to accept.")
            print("(or 'q' to quit)")
            while True:
                ch2 = getch_blocking()
                if not ch2:
                    continue
                c2 = ch2.lower()
                if c2 == "q":
                    clean_before_quit(voice_name)
                    print("👋 Exiting as requested.")
                    sys.exit(0)
                elif c2 == "p":
                    dur = wav_duration_seconds(tmp_path)
                    print(f"\n▶ Playing ( {dur:.2f}s, {SR} Hz, mono, {SUBTYPE} )")
                    play_wav(tmp_path)
                    print("\nPress 'P' to play again ▶")
                    print("'D' to delete and re-record ⏺")
                    print("'A' to accept.")
                    print("(or 'q' to quit)")
                    continue
                elif c2 == "d":
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except Exception as e:
                            print(f"⚠️ Could not delete temp file: {e}")
                    print("⏹ Last recording deleted ⏹")
                    print("▶ Press any key to re-start recording ⏺")
                    ch3 = getch_blocking()
                    if ch3 and ch3.lower() == "q":
                        clean_before_quit(voice_name)
                        print("👋 Exiting. Goodbye!")
                        sys.exit(0)
                    return "repeat"
                elif c2 == "a":
                    return "accepted"
                else:
                    # ignore unrecognized and re-prompt secondary menu
                    continue

        elif c == "a":
            return "accepted"

        else:
            # Ignore and re-prompt the primary menu
            continue

def clean_audio(in_path, out_path):
    """Apply noise reduction + normalization."""
    print("Cleaning audio...")

    # Load raw audio
    data, sr = sf.read(in_path)

    # Apply noise reduction
    reduced = nr.reduce_noise(y=data, sr=sr)

    # Save temporary cleaned file
    tmp_path = out_path.replace(".wav", "_cleaning.wav")
    sf.write(tmp_path, reduced, sr)

    # Normalize with pydub
    audio = AudioSegment.from_wav(tmp_path)
    audio = effects.normalize(audio)

    audio.export(out_path, format="wav")
    os.remove(tmp_path)

    print(f"Cleaned audio saved: {out_path}")

# -----------------------------
# Finalize helper (new)
# -----------------------------
def finalize_voice_identity(voice_name: str):
    """
    Run cleaning on <voice_name>.wav -> clean_<voice_name>.wav and delete the original.
    Exits the program on success.
    """
    main_path = ROOT_DIR / f"{voice_name}.wav"
    out_path  = ROOT_DIR / f"clean_{voice_name}.wav"

    # Ensure main has audio
    if not main_path.exists():
        print("⚠️ No main voice file found to finalize.")
        return
    try:
        info = sf.info(str(main_path))
        if info.frames == 0:
            print("⚠️ Your main voice file is empty; please record and accept at least one take before finalizing.")
            return
    except Exception:
        if main_path.stat().st_size <= 44:
            print("⚠️ Your main voice file appears empty; please record and accept at least one take before finalizing.")
            return

    # Clean & normalize, then delete original
    clean_audio(str(main_path), str(out_path))
    try:
        main_path.unlink()
    except Exception as e:
        print(f"⚠️ Could not delete original file after cleaning: {e}")

    print(f"✅ Final voice identity saved as: {out_path.name}")
    # print("👋 Exiting. Goodbye!")
    sys.exit(0)

def main():
    global CURRENT_VOICE_NAME
    # 1) Greeting & name prompt
    print_banner()
    voice_name = ask_for_voice_identity_name()  # without .wav
    CURRENT_VOICE_NAME = voice_name
    main_path = ROOT_DIR / f"{voice_name}.wav"
    tmp_path = ROOT_DIR / f"{voice_name}_tmp.wav"

    # 3) Create empty main and temp WAVs
    create_empty_wav(main_path)
    create_empty_wav(tmp_path)

    # 4) Ready prompt
    print("\nSystem is ready to create your voice identity.")
    print("Create by recording your voice for 5-10 minutes in your relevant field.")
    print("Record in a noise free room.")
    prompt_press_any_key_or_quit_line(voice_name)

    # Loop for record → review → accept → enrich
    while True:
        # 5) Record to tmp until keypress stop
        duration = record_to_file(tmp_path)

        # 6) Post-record menus (P/D/A/q)
        action = prompt_review_menu(tmp_path, voice_name, duration)

        if action == "repeat":
            # back to recording
            continue

        if action == "accepted":
            # 9.5) Append tmp into main, delete tmp
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                print("⚠️ Nothing to append; please record again.")
                continue
            append_wav(main_path, tmp_path)
            try:
                tmp_path.unlink()
            except Exception as e:
                print(f"⚠️ Could not delete temp file after append: {e}")

            final_dur = wav_duration_seconds(main_path)
            print(f"\n✅ Appended to {main_path.name}.")
            print(f"Current {voice_name} length: {final_dur:.2f}s ( {SR} Hz, mono, {SUBTYPE} )")

            # 10) Ask to enrich OR SUBMIT FINAL
            print(f"\nEnrich your voice identity ({voice_name}) by recording more,")
            print("press 'S' to submit the final voice identity now, or press 'q+↵' to quit.")
            print("▶ Press any key to start recording ⏺")

            # Handle S / q / any key (start recording)
            ch = getch_blocking()
            if ch and ch.lower() == "q":
                clean_before_quit(voice_name)
                print("👋 Exiting. Goodbye!")
                sys.exit(0)
            if ch and ch.lower() == "s":
                # finalize_voice_identity(voice_name)
                ####### Need to remove voice_tmp.wav and quit.
                print("👋 Exiting. Goodbye!")

            # Recreate an empty temp file for the next take, then continue to next loop (which starts recording)
            create_empty_wav(tmp_path)
            # 11) Repeat from step 5 (loop top)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clean_before_quit(CURRENT_VOICE_NAME)
        print("\n👋 Exiting. Stay awesome!")
