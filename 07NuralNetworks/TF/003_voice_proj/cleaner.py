# cleaner.py
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

DATA_DIR = Path("./data")
CLEAN_DIR = DATA_DIR / "clean"

def clean_audio_preserve_format(input_path: Path, output_path: Path):
    # Read original metadata
    with sf.SoundFile(input_path, mode="r") as f:
        orig_sr = f.samplerate
        orig_channels = f.channels
        orig_subtype = f.subtype   # e.g., "PCM_16"
        orig_format = f.format     # e.g., "WAV"

    # Load as float32 for processing; keep channels as 2D
    data, sr = sf.read(input_path, dtype="float32", always_2d=True)
    if sr != orig_sr:
        print(f"⚠️ Warning: detected samplerate mismatch {sr} vs header {orig_sr}. Using header value.")
        sr = orig_sr

    # Noise reduction per channel
    cleaned = np.empty_like(data)
    for ch in range(data.shape[1]):
        cleaned[:, ch] = nr.reduce_noise(y=data[:, ch], sr=sr)

    # Prepare for writing with original subtype
    if (orig_subtype or "").upper() == "PCM_16":
        cleaned_to_write = np.clip(cleaned * 32768.0, -32768, 32767).astype(np.int16)
    else:
        cleaned_to_write = np.clip(cleaned, -1.0, 1.0).astype(np.float32)

    # Write preserving original container, subtype, and sample rate
    sf.write(
        file=output_path,
        data=cleaned_to_write,
        samplerate=orig_sr,
        subtype=orig_subtype,
        format=orig_format,
    )

    print(f"✅ Cleaned file saved: {output_path}")
    print(f"   • Format: {orig_format} | Subtype: {orig_subtype} | SR: {orig_sr} | Channels: {orig_channels}")
    # Return float32 arrays in [-1, 1] for plotting
    return data, cleaned, orig_sr

def resolve_input(arg: str) -> Path:
    p = Path(arg)
    if not p.parent or str(p.parent) in {".", ""}:
        p = DATA_DIR / p.name
    return p

def ask_yes_no(prompt: str, default_no: bool = True) -> bool:
    ans = input(prompt).strip().lower()
    if ans in {"y", "yes"}:
        return True
    if ans in {"n", "no"}:
        return False
    return not default_no

def parse_time_range(spec: str, total_seconds: float):
    """
    Accepts:
      - 't1:t2'  -> (t1, t2)
      - 't1'     -> (0, t1)
      - ':t2'    -> (0, t2)
      - 't1:'    -> (t1, total_seconds)
    Returns (t1, t2) clamped to [0, total_seconds], ensuring t2 > t1.
    Raises ValueError on invalid input.
    """
    s = (spec or "").strip().replace(" ", "")
    if not s:
        return 0.0, total_seconds

    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError("Invalid time range format.")
        left, right = parts[0], parts[1]
        t1 = float(left) if left != "" else 0.0
        t2 = float(right) if right != "" else total_seconds
    else:
        t1, t2 = 0.0, float(s)

    # Clamp to [0, total_seconds]
    t1 = max(0.0, min(t1, total_seconds))
    t2 = max(0.0, min(t2, total_seconds))
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1 and within track length.")
    return t1, t2

def get_screen_figure_size():
    """
    Try to make the plot as wide as the screen.
    Returns (width_inches, height_inches).
    """
    default = (18.0, 5.0)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()
        dpi = plt.rcParams.get("figure.dpi", 100)
        root.destroy()
        width_in = max(12.0, width_px / dpi)
        height_in = max(4.0, min(0.35 * (height_px / dpi), 8.0))
        return (width_in, height_in)
    except Exception:
        return default

def maximize_window_if_possible():
    """Try to maximize the figure window across common backends."""
    try:
        mgr = plt.get_current_fig_manager()
        # TkAgg:
        try:
            mgr.window.state('zoomed')
        except Exception:
            pass
        # Qt5Agg:
        try:
            mgr.window.showMaximized()
        except Exception:
            pass
    except Exception:
        pass

def visualize_with_range(original_f32_2d: np.ndarray, cleaned_f32_2d: np.ndarray, sr: int):
    """
    Returns the chosen (t1, t2) time span used for plotting.
    """
    # Mono mix for plotting only
    orig = original_f32_2d.mean(axis=1)
    clean = cleaned_f32_2d.mean(axis=1)

    total_seconds = len(orig) / sr
    print(f"Audio length: {total_seconds:.3f} s")

    # Ask for time range using the flexible input
    while True:
        spec = input(
            "Enter time range to plot (e.g., '1.5:3.2', '2:', ':5', '4.0', or leave blank for full): "
        )
        try:
            t1, t2 = parse_time_range(spec, total_seconds)
            break
        except Exception as e:
            print(f"Invalid range: {e}")

    # Precompute time array for entire signal (lazy slice later)
    t = np.arange(len(orig)) / sr
    window_dur = t2 - t1

    # Figure size attempts to match screen width
    w_in, h_in = get_screen_figure_size()
    fig, ax = plt.figure(figsize=(w_in, h_in)), plt.gca()
    maximize_window_if_possible()

    # Initial slice
    start_idx = int(t1 * sr)
    end_idx = int(t2 * sr)
    line_orig, = ax.plot(t[start_idx:end_idx], orig[start_idx:end_idx], label="Original")
    line_clean, = ax.plot(t[start_idx:end_idx], clean[start_idx:end_idx], label="Cleaned", alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Original vs Cleaned  |  Range: {t1:.3f}–{t2:.3f} s")
    ax.legend()
    ax.set_xlim(t1, t2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave room for slider

    # Horizontal scroll slider when window < full duration
    if window_dur < total_seconds:
        slider_ax = fig.add_axes([0.10, 0.02, 0.80, 0.03])
        s_max = max(0.0, total_seconds - window_dur)
        scroll = Slider(
            ax=slider_ax,
            label="Scroll (s)",
            valmin=0.0,
            valmax=s_max,
            valinit=t1,
            valstep=window_dur / 200.0 if window_dur > 0 else 0.01
        )

        def on_scroll(val):
            start = scroll.val
            end = start + window_dur
            si = int(start * sr)
            ei = int(end * sr)
            line_orig.set_data(t[si:ei], orig[si:ei])
            line_clean.set_data(t[si:ei], clean[si:ei])
            ax.set_xlim(start, end)
            ax.set_title(f"Original vs Cleaned  |  Range: {start:.3f}–{end:.3f} s")
            # Rescale y-limits to visible data
            try:
                ymin = min(np.min(orig[si:ei]), np.min(clean[si:ei]))
                ymax = max(np.max(orig[si:ei]), np.max(clean[si:ei]))
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                    pad = 0.05 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
            except Exception:
                pass
            fig.canvas.draw_idle()

        scroll.on_changed(on_scroll)

    plt.show()
    return t1, t2

def compute_fft_db(x: np.ndarray, sr: int):
    """
    Compute single-sided magnitude spectrum in dBFS using rFFT.
    Returns (freqs_hz, mag_db).
    """
    N = len(x)
    if N < 2:
        return np.array([0.0]), np.array([-120.0])

    # Hann window to reduce spectral leakage
    win = np.hanning(N)
    xw = x * win

    # rFFT and frequency bins
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)

    # Amplitude correction for Hann window and single-sided spectrum
    # Scale by sum(win)/2 to approximately preserve amplitudes
    scale = np.sum(win) / 2.0
    mag = np.abs(X) / (scale + 1e-12)

    # Avoid log(0)
    mag_db = 20.0 * np.log10(mag + 1e-12)
    return freqs, mag_db

def plot_spectrum_for_range(original_f32_2d: np.ndarray, cleaned_f32_2d: np.ndarray, sr: int, t1: float, t2: float):
    """
    Plot magnitude spectra (dB) for original and cleaned signals over [t1, t2].
    Uses mono-mix for analysis only; saved audio remains unchanged.
    """
    orig = original_f32_2d.mean(axis=1)
    clean = cleaned_f32_2d.mean(axis=1)

    si = int(max(0.0, t1) * sr)
    ei = int(max(t1, t2) * sr)
    segment_o = orig[si:ei]
    segment_c = clean[si:ei]

    # Remove DC (mean) to focus on spectral content
    segment_o = segment_o - np.mean(segment_o) if segment_o.size else segment_o
    segment_c = segment_c - np.mean(segment_c) if segment_c.size else segment_c

    f_o, db_o = compute_fft_db(segment_o, sr)
    f_c, db_c = compute_fft_db(segment_c, sr)

    w_in, h_in = get_screen_figure_size()
    plt.figure(figsize=(w_in, h_in))
    maximize_window_if_possible()

    plt.plot(f_o, db_o, label="Original (dB)")
    plt.plot(f_c, db_c, label="Cleaned (dB)", alpha=0.85)
    plt.xlim(0, sr / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Spectrum {t1:.3f}–{t2:.3f} s (Hann windowed)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python ./cleaner.py <input-audio-filename-or-path>")
        print("Example: python ./cleaner.py my_speech.wav  (reads ./data/my_speech.wav)")
        sys.exit(1)

    input_arg = sys.argv[1]
    input_path = resolve_input(input_arg)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLEAN_DIR / input_path.name

    # Overwrite confirmation
    if output_path.exists():
        ans = input(f"⚠️ {output_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("🚫 Aborted without overwriting.")
            sys.exit(0)

    try:
        orig_f32_2d, clean_f32_2d, sr = clean_audio_preserve_format(input_path, output_path)
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        sys.exit(1)

    # Optional waveform visualization
    if ask_yes_no("Do you want to visualize original vs cleaned? [y/N]: ", default_no=True):
        try:
            t1, t2 = visualize_with_range(orig_f32_2d, clean_f32_2d, sr)
        except Exception as e:
            print(f"⚠️ Could not plot waveform: {e}")
            t1 = t2 = None
    else:
        t1 = t2 = None

    # Next: Frequency-domain spectrum for that chosen span
    if t1 is not None and t2 is not None:
        if ask_yes_no("Do you want the frequency-domain spectrum for that time span? [y/N]: ", default_no=True):
            try:
                plot_spectrum_for_range(orig_f32_2d, clean_f32_2d, sr, t1, t2)
            except Exception as e:
                print(f"⚠️ Could not plot spectrum: {e}")

if __name__ == "__main__":
    main()
