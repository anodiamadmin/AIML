# pip install sounddevice soundfile
import sounddevice as sd
import soundfile as sf
import sys, os, threading, queue, time

SR = 16000          # or 22050
CHANNELS = 1
SUBTYPE = "PCM_16"  # 16-bit PCM
OUT_PATH = "./data/user_voice.wav"

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
    # Wait for keypress, then set the event
    getch_blocking()
    event.set()

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("Press any key to START recording…")
    getch_blocking()

    print("Recording… Press any key to STOP.")
    stop_event = threading.Event()
    threading.Thread(target=keypress_to_event, args=(stop_event,), daemon=True).start()

    q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # Ensure 16-bit int data is queued
        q.put(indata.copy())

    frames_written = 0
    start_time = time.time()

    try:
        with sf.SoundFile(OUT_PATH, mode="w", samplerate=SR, channels=CHANNELS, subtype=SUBTYPE) as wav_file:
            with sd.InputStream(samplerate=SR, channels=CHANNELS, dtype='int16', callback=audio_callback):
                while not stop_event.is_set():
                    data = q.get()  # blocks until a chunk arrives
                    wav_file.write(data)
                    frames_written += len(data)

                # Drain any buffered chunks after stop
                while not q.empty():
                    data = q.get_nowait()
                    wav_file.write(data)
                    frames_written += len(data)

    except KeyboardInterrupt:
        print("\nInterrupted. Finalizing file…")

    duration = frames_written / float(SR)
    print(f"Saved {OUT_PATH}  ({duration:.1f}s, {SR} Hz, mono, {SUBTYPE})")

if __name__ == "__main__":
    main()
