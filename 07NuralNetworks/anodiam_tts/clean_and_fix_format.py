#!/usr/bin/env python3
"""
clean_and_fix_format.py  (numba-free)

Usage:
  python clean_and_fix_format.py .\data\input_speech\abcd.mp3
  python clean_and_fix_format.py -i .\data\model_voices\voice_id_1.wav
  python clean_and_fix_format.py .\data\input_speech\abcd.mp3 --save-plot .\data\input_speech\clean_abcd_spectrum.png

What it does:
  1) Uses fix_format.py to ensure a processing copy at WAV mono @16k PCM_16 (temp if needed).
  2) Applies spectral-subtraction denoising (SciPy STFT/ISTFT).
  3) Saves cleaned audio next to the original as 'clean_<basename>.wav' (WAV mono @16k PCM_16).
  4) Plots overlapped frequency spectra (raw vs cleaned); optional --save-plot.

Deps:
  pip install numpy soundfile scipy matplotlib
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, get_window

# --- Import conversion/utility logic from fix_format.py (DO NOT duplicate) ---
try:
    from fix_format import (
        check_ffmpeg_tools,
        needs_conversion,
        convert,
        unique_out_path,
        REQUIRED_SR,   # 16000
        REQUIRED_CH,   # 1
    )
except Exception as e:
    print("ERROR: Could not import from fix_format.py. Make sure it exists in the same folder.", file=sys.stderr)
    raise

# --- Denoise params (conservative defaults) ---
N_FFT = 1024
HOP = 256
WIN = 1024
WINDOW = "hann"
NOISE_PERCENTILE = 10.0    # estimate noise floor at this percentile per frequency bin
OVER_SUB = 1.2             # over-subtraction factor for noise magnitude
SPECTRAL_FLOOR = 0.02      # minimum fraction of original magnitude (avoid musical noise)

# --- Spectrum viz params ---
SPEC_N_FFT = 2048
EPS = 1e-12


def prepare_processing_wav(in_path: Path) -> Path:
    """
    Ensure we have a WAV mono @16k PCM_16 to work on.
    - If input already matches required format, return original path.
    - Else, convert to a temporary .wav (using fix_format.convert) and return that path.
    """
    check_ffmpeg_tools()
    if not needs_conversion(in_path):
        return in_path

    tmpdir = tempfile.mkdtemp(prefix="cleaner_fmt_")
    tmp_wav = Path(tmpdir) / (in_path.stem + "_16k_mono.wav")
    convert(in_path, tmp_wav)
    return tmp_wav


def spectral_subtraction_denoise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple spectral subtraction using a percentile noise estimate per frequency bin.
    Uses SciPy STFT/ISTFT (no numba/llvmlite).
    """
    win = get_window(WINDOW, WIN, fftbins=True)
    f, t, Zxx = stft(y, fs=sr, window=win, nperseg=WIN, noverlap=WIN - HOP, nfft=N_FFT, boundary='zeros', padded=True)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Noise estimate (per frequency bin)
    noise_mag = np.percentile(mag, NOISE_PERCENTILE, axis=1)  # shape (freq_bins,)

    # Over-subtract and apply a spectral floor
    mag_d = np.maximum(mag - OVER_SUB * noise_mag[:, None], SPECTRAL_FLOOR * mag)

    Zxx_d = mag_d * np.exp(1j * phase)
    _, y_d = istft(Zxx_d, fs=sr, window=win, nperseg=WIN, noverlap=WIN - HOP, input_onesided=True, boundary=True)

    # Match original length (istft can differ by a few samples due to padding)
    if len(y_d) != len(y):
        y_d = y_d[:len(y)] if len(y_d) > len(y) else np.pad(y_d, (0, len(y) - len(y_d)))

    y_d = np.clip(y_d, -1.0, 1.0).astype(np.float32)
    return y_d


def compute_avg_spectrum(y: np.ndarray, sr: int):
    """
    Compute an average magnitude spectrum over time via STFT power; return (freqs_hz, mag_db_norm).
    """
    win = get_window(WINDOW, SPEC_N_FFT, fftbins=True)
    f, t, Z = stft(y, fs=sr, window=win, nperseg=SPEC_N_FFT, noverlap=SPEC_N_FFT // 4, nfft=SPEC_N_FFT, boundary='zeros', padded=True)
    P = np.abs(Z) ** 2
    p_avg = np.mean(P, axis=1) + EPS
    # Normalize to 0 dB max for clearer overlap
    mag_db = 10 * np.log10(p_avg / np.max(p_avg))
    return f, mag_db


def plot_spectra(raw_y: np.ndarray, clean_y: np.ndarray, sr: int, title: str, save_path: Path = None):
    """
    Plot overlapped frequency spectra: raw vs cleaned.
    """
    freqs_raw, mag_db_raw = compute_avg_spectrum(raw_y, sr)
    freqs_clean, mag_db_clean = compute_avg_spectrum(clean_y, sr)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs_raw, mag_db_raw, label="Raw", linewidth=1.6, color="#ff0000", alpha=1.0)  # bright red, opaque
    plt.plot(freqs_clean, mag_db_clean, label="Cleaned", linewidth=1.6, color="#00ff00",
             alpha=0.5)  # bright green, 50% transparent
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, normalized)")
    plt.xlim([0, sr / 2])
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    if save_path is not None:
        save_path = save_path.with_suffix(".png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved spectrum plot: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Denoise audio and save as 'clean_<basename>.wav' (WAV mono @16k PCM_16). Overlays raw vs cleaned spectra."
    )
    # Accept multiple flag styles and an optional positional path
    parser.add_argument("-i", "-input", "--input", dest="input_path",
                        help="Path to input audio file (you can also pass it positionally).")
    parser.add_argument("positional_path", nargs="?", help="Input audio file (positional)")
    parser.add_argument("--save-plot", dest="save_plot", default=None,
                        help="Optional path to save the spectrum PNG (e.g., ./data/input_speech/clean_abcd_spectrum.png)")

    args = parser.parse_args()
    in_arg = args.input_path or args.positional_path
    if not in_arg:
        parser.error("Provide an input file via --input <path> or as a positional argument.")

    in_path = Path(in_arg).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure we have a 16k mono WAV to process (via fix_format)
    processing_wav = prepare_processing_wav(in_path)

    # Load raw processing signal (already 16k mono PCM_16)
    raw_y, sr = sf.read(str(processing_wav), dtype="float32", always_2d=False)
    if raw_y.ndim > 1:
        raw_y = np.mean(raw_y, axis=1)  # safety; should already be mono
    if sr != REQUIRED_SR:
        # Shouldn't happen because fix_format enforces 16k; just in case:
        from scipy.signal import resample_poly
        raw_y = resample_poly(raw_y, up=REQUIRED_SR, down=sr)
        sr = REQUIRED_SR

    # Denoise
    clean_y = spectral_subtraction_denoise(raw_y, sr)

    # Output path: same folder as ORIGINAL input, prefix 'clean_', always .wav
    out_dir = in_path.parent
    out_name = f"clean_{in_path.stem}.wav"
    out_path = unique_out_path(out_dir / out_name)

    # Save as WAV mono @16k PCM_16
    sf.write(str(out_path), clean_y, REQUIRED_SR, subtype="PCM_16")
    print(f"Saved cleaned audio: {out_path}")

    # Plot spectra (raw vs cleaned)
    title = f"Frequency Spectrum: {in_path.name} (raw) vs {out_path.name} (cleaned)"
    save_path = Path(args.save_plot).expanduser().resolve() if args.save_plot else None
    plot_spectra(raw_y, clean_y, sr, title=title, save_path=save_path)


if __name__ == "__main__":
    main()
