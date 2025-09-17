# pip install sounddevice soundfile
import sounddevice as sd
import soundfile as sf
import sys, os, threading, queue, time

SR = 16000          # or 22050
CHANNELS = 1
SUBTYPE = "PCM_16"  # 16-bit PCM
OUT_DIR = "./data"  # folder where all takes are stored

def getch_blocking():
    """Wait for a single keypress (any key), cross-platform."""
    try:
        # Windows
        import msvcrt
        msvcrt.getch()
    except ImportError:
        # macOS / Linux
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

def keypress_to_event(event):
    getch_blocking()
    event.set()

def record_take(filename):
    """Record one take and save as <filename>.wav"""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not filename.endswith(".wav"):
        filename += ".wav"
    out_path = os.path.join(OUT_DIR, filename)

    print(f"\nPress any key to START recording: {filename}")
    getch_blocking()

    print("Recording… Press any key to STOP.")
    stop_event = threading.Event()
    threading.Thread(target=keypress_to_event, args=(stop_event,), daemon=True).start()

    q = queue.Queue()
    frames_written = 0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        with sf.SoundFile(out_path, mode="w", samplerate=SR,
                          channels=CHANNELS, subtype=SUBTYPE) as wav_file:
            with sd.InputStream(samplerate=SR, channels=CHANNELS,
                                dtype='int16', callback=audio_callback):
                while not stop_event.is_set():
                    data = q.get()
                    wav_file.write(data)
                    frames_written += len(data)

                while not q.empty():
                    data = q.get_nowait()
                    wav_file.write(data)
                    frames_written += len(data)
    except KeyboardInterrupt:
        print("\nInterrupted. Finalizing file…")

    duration = frames_written / float(SR)
    print(f"✅ Saved {out_path}  ({duration:.1f}s, {SR} Hz, mono, {SUBTYPE})")

def main():
    while True:
        filename = input("\nEnter filename in the following format 'your_voice_clip' for this take (without extension, or 'q + ENTER' to quit): ").strip()
        if filename.lower() == "q":
            print("Bye! Recording session finished.")
            break
        if not filename:
            print("⚠️  Filename cannot be empty!")
            continue

        record_take(filename)

if __name__ == "__main__":
    main()
