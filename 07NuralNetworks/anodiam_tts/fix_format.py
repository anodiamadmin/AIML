# Example
# python fix_format.py -input=./data/input_speech/acd.mp3
# python fix_format.py --input ./data/input_speech/acd.wav
# python .\fix_format.py --input .\data\input_speech\file_example_MP3.mp3
# python .\fix_format.py -i .\data\input_speech\file_example_MP3.mp3
# python .\fix_format.py .\data\input_speech\file_example_MP3.mp3
# or
# python .\fix_format.py --input=".\data\input_speech\file_example_MP3.mp3"

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_SR = 16000
REQUIRED_CH = 1
REQUIRED_FMT = "s16"  # PCM 16-bit
REQUIRED_CODEC = "pcm_s16le"  # explicit encoder for WAV

def run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as e:
        # Return the object so caller can inspect stderr
        return e

def check_ffmpeg_tools():
    for tool in ("ffmpeg", "ffprobe"):
        r = run([tool, "-version"])
        if r is None:
            print(f"ERROR: '{tool}' not found. Install FFmpeg and ensure it’s on your PATH.", file=sys.stderr)
            sys.exit(1)

def ffprobe_audio_info(path: Path):
    """Return (format_name, sample_rate:int, channels:int, sample_fmt:str) or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=format_name",
        "-show_entries", "stream=codec_name,channels,sample_rate,sample_fmt",
        "-select_streams", "a:0",
        "-of", "json",
        str(path)
    ]
    r = run(cmd)
    if r is None or (r.returncode not in (0, None)):
        # mp3 or odd files may still be probeable; if not, we’ll just convert blindly
        return None
    data = json.loads(r.stdout)
    fmt = (data.get("format") or {}).get("format_name", "")
    streams = data.get("streams") or []
    if not streams:
        return fmt, None, None, None
    s = streams[0]
    try:
        sr = int(s.get("sample_rate")) if s.get("sample_rate") else None
    except Exception:
        sr = None
    ch = s.get("channels")
    sfmt = s.get("sample_fmt")
    return fmt, sr, ch, sfmt

def needs_conversion(path: Path):
    """Decide if we must convert to WAV 16k mono PCM_16."""
    info = ffprobe_audio_info(path)
    if info is None:
        # Could not probe; safest to convert
        return True
    fmt_name, sr, ch, sfmt = info
    is_wav_container = isinstance(fmt_name, str) and ("wav" in fmt_name.lower())
    return not (is_wav_container and sr == REQUIRED_SR and ch == REQUIRED_CH and sfmt == REQUIRED_FMT)

def unique_out_path(desired: Path) -> Path:
    """Return a non-existing path by appending (1), (2), … if needed."""
    if not desired.exists():
        return desired
    stem = desired.stem
    suffix = desired.suffix
    parent = desired.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def build_output_path(in_path: Path, will_convert: bool) -> Path:
    # If the source is not .wav or we must convert, prefer <base>.wav
    # If it’s already a .wav but needs conversion, write <base>_fixed.wav
    base = in_path.with_suffix("")  # drop extension
    if in_path.suffix.lower() != ".wav":
        out = base.with_suffix(".wav")
    else:
        # Same extension; avoid overwriting original
        out = in_path.with_name(f"{in_path.stem}_fixed.wav")
    return unique_out_path(out)

def convert(in_path: Path, out_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ar", str(REQUIRED_SR),
        "-ac", str(REQUIRED_CH),
        "-c:a", REQUIRED_CODEC,   # ensures WAV PCM 16-bit little endian
        str(out_path)
    ]
    r = run(cmd)
    if r is None:
        print("ERROR: ffmpeg not found.", file=sys.stderr)
        sys.exit(1)
    if isinstance(r, subprocess.CalledProcessError):
        print("ERROR during conversion:\n", r.stderr, file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Ensure audio is WAV mono 16k PCM_16; convert if needed."
    )
    # Accept multiple flag styles AND an optional positional path
    parser.add_argument("-i", "-input", "--input", dest="input_path",
                        help="Path to input audio file (you can also pass it positionally).")
    parser.add_argument("positional_path", nargs="?", help="Input audio file (positional)")

    args = parser.parse_args()
    in_arg = args.input_path or args.positional_path
    if not in_arg:
        parser.error("Please provide an input file via --input <path> or as a positional argument.")

    in_path = Path(in_arg).expanduser().resolve()

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure ffmpeg/ffprobe available
    check_ffmpeg_tools()

    # Decide whether conversion is required
    must_convert = needs_conversion(in_path)

    if not must_convert:
        if in_path.suffix.lower() == ".wav":
            print(f"OK: Already WAV mono @{REQUIRED_SR} Hz PCM_16 -> {in_path}")
            sys.exit(0)
        else:
            # Audio params are fine but container isn't WAV; convert to WAV container
            print("Container is not WAV; converting to WAV with the same required parameters.")

    out_path = build_output_path(in_path, True)
    convert(in_path, out_path)

    # Verify output
    info = ffprobe_audio_info(out_path)
    if info is None:
        print(f"WARNING: Could not verify output precisely. File saved to: {out_path}")
        sys.exit(0)

    fmt, sr, ch, sfmt = info
    ok = (fmt and "wav" in fmt.lower()
          and sr == REQUIRED_SR and ch == REQUIRED_CH and sfmt == REQUIRED_FMT)

    if ok:
        print(f"SUCCESS: {out_path} is WAV mono @{REQUIRED_SR} Hz PCM_16")
    else:
        print(f"WARNING: Output saved to {out_path} but verification was inconclusive.")


if __name__ == "__main__":
    main()
