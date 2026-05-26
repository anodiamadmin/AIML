#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rvc_trainer.py — minimal wrapper to train an RVC speaker model from a single WAV file.

Usage:
    python rvc_trainer.py -i="voice_id_x.wav" -e=10 -c=2

Behavior:
- Expects input at ./data/model_voices/<file_name>, e.g., ./model_voices/voice_id_x.wav
- Stages data into ./rvc/datasets/<voice_id>/
- Calls ./rvc/infer/modules/train/train.py with:
    -e <voice_id>  -sr 48k  -f0 0  -bs 4  -te <n>  -se <a>  -pg <pretrained G>  -sw 1  -v v1  -l 1  -c 0
- After training completes, copies:
    - latest G_*.pth  → ./rvc/models/<voice_id>.pth
    - every G_*.pth   → ./rvc/models/<voice_id>_G_<step>.pth

Notes:
- This wrapper doesn’t do dataset cleaning or phoneme balancing; it just stages your WAV.
- If you need F0 training (singing / expressive speech), change F0_ON below to 1.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# -------- Project-relative paths --------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_VOICES_DIR = PROJECT_ROOT / "data" / "model_voices"
RVC_ROOT = PROJECT_ROOT / "rvc"
RVC_DATASETS = RVC_ROOT / "datasets"
RVC_LOGS = RVC_ROOT / "logs"
RVC_MODELS_OUT = RVC_ROOT / "models"

TRAIN_SCRIPT = RVC_ROOT / "infer" / "modules" / "train" / "train.py"

# Common pretrained candidates to try in order:
PRETRAINED_G_CANDIDATES = [
    RVC_ROOT / "assets" / "pretrained" / "G48K.pth",   # 48k model (typical)
    RVC_ROOT / "assets" / "pretrained" / "G40K.pth",   # 40k fallback
]

# ----- Default training knobs you can tweak -----
SAMPLE_RATE_FLAG = "48k"   # "40k" or "48k"
F0_ON = 0                  # 0 = off, 1 = on
BATCH_SIZE = 4             # safe default; raise if you have VRAM
VERSION = "v1"             # RVC version flag
SAVE_WEIGHTS = 1           # 1 = save model weights
LOG_MULTI_SPEAKER = 1      # '-l' flag; keep at 1 (multi speaker log format)
CACHE_IN_GPU = 0           # '-c' flag; usually 0 is fine


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def pick_pretrained_g():
    for p in PRETRAINED_G_CANDIDATES:
        if p.is_file():
            return p
    return None


def ensure_paths():
    for p in [RVC_DATASETS, RVC_LOGS, RVC_MODELS_OUT]:
        p.mkdir(parents=True, exist_ok=True)


def slug_from_filename(filename: str) -> str:
    """
    Turn 'voice_id_x.wav' into 'voice_id_x'
    """
    base = Path(filename).name
    stem = Path(base).stem
    # Require snake_case-ish: letters, numbers, underscores
    if not re.fullmatch(r"[A-Za-z0-9_]+", stem):
        die(f"Input file stem must be alphanumeric/underscore only (got '{stem}').")
    return stem


def stage_wav(input_wav: Path, voice_id: str):
    """
    Copy the input wav into the RVC datasets folder
    under datasets/<voice_id>/wavs/.
    """
    target_dir = RVC_DATASETS / voice_id / "wavs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / input_wav.name

    shutil.copy2(input_wav, target_path)
    print(f"✅ Staged {input_wav} → {target_path}")
    return target_path


def run_rvc_train(voice_id: str, total_epochs: int, save_every: int, pretrained_g: Path):
    if not TRAIN_SCRIPT.is_file():
        die(f"RVC train script not found at: {TRAIN_SCRIPT}\n"
            f"Make sure the RVC repo exists at ./rvc and has the expected structure.")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "-e", voice_id,
        "-sr", SAMPLE_RATE_FLAG,
        "-f0", str(F0_ON),
        "-bs", str(BATCH_SIZE),
        "-te", str(total_epochs),
        "-se", str(save_every),
        "-pg", str(pretrained_g),
        "-sw", str(SAVE_WEIGHTS),
        "-v", VERSION,
        "-l", str(LOG_MULTI_SPEAKER),
        "-c", str(CACHE_IN_GPU),
    ]

    print(">> Running RVC training:")
    print("   ", " ".join([f'"{c}"' if " " in c else c for c in cmd]))
    print()

    # Ensure imports like "from infer.lib..." work:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RVC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, check=True, cwd=str(RVC_ROOT), env=env)
    except subprocess.CalledProcessError as e:
        die(f"Training process exited with status {e.returncode}. See console/logs for details.")


def collect_checkpoints(voice_id: str):
    """
    Find all G_*.pth under ./rvc/logs/<voice_id>/ and copy:
      - every checkpoint → ./rvc/models/<voice_id>_G_<step>.pth
      - latest checkpoint → ./rvc/models/<voice_id>.pth (canonical name)
    """
    exp_dir = RVC_LOGS / voice_id
    if not exp_dir.exists():
        die(f"Expected logs directory not found: {exp_dir}")

    ckpt_files = sorted(exp_dir.glob("G_*.pth"))

    if not ckpt_files:
        # Some forks save under weights/ or elsewhere; try a broader search:
        ckpt_files = sorted(exp_dir.rglob("G_*.pth"))

    if not ckpt_files:
        die(f"No G_*.pth checkpoints found under {exp_dir}. Did training complete?")

    # Copy all checkpoints with detailed names
    for ck in ckpt_files:
        step_match = re.search(r"G_(\d+)\.pth$", ck.name)
        step_suffix = step_match.group(1) if step_match else "unknown"
        dst_named = RVC_MODELS_OUT / f"{voice_id}_G_{step_suffix}.pth"
        shutil.copy2(ck, dst_named)
        print(f"Saved checkpoint copy: {dst_named}")

    # Latest (by step in filename if possible, else by mtime)
    def step_number(p: Path):
        m = re.search(r"G_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    ckpt_sorted = sorted(ckpt_files, key=lambda p: (step_number(p), p.stat().st_mtime))
    latest = ckpt_sorted[-1]
    canonical = RVC_MODELS_OUT / f"{voice_id}.pth"
    shutil.copy2(latest, canonical)
    print(f"\nFinal model saved as: {canonical}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train an RVC speaker model from a single WAV and save checkpoints."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input WAV filename (located under ./model_voices/), e.g., voice_id_x.wav")
    parser.add_argument("-e", "--epochs", required=True, type=int,
                        help="Total epochs (n).")
    parser.add_argument("-c", "--checkpoint_every", required=True, type=int,
                        help="Save checkpoint every 'a' epochs (before the nth epoch).")

    args = parser.parse_args()

    # Resolve paths and voice_id
    ensure_paths()
    input_name = args.input.strip().strip('"').strip("'")
    if not input_name.lower().endswith(".wav"):
        die("Input must be a .wav file name (e.g., voice_id_x.wav)")

    voice_id = slug_from_filename(input_name)

    src_wav = MODEL_VOICES_DIR / input_name
    if not src_wav.is_file():
        die(f"Input WAV not found at: {src_wav}")

    if args.epochs <= 0:
        die("Epochs (-e) must be a positive integer.")
    if args.checkpoint_every <= 0 or args.checkpoint_every >= args.epochs:
        die("Checkpoint interval (-c) must be > 0 and < total epochs (-e).")

    pretrained_g = pick_pretrained_g()
    if pretrained_g is None:
        die("Could not find a pretrained generator. "
            "Expected one of:\n  - ./rvc/assets/pretrained/G48K.pth\n  - ./rvc/assets/pretrained/G40K.pth\n"
            "Please download the appropriate file and try again.")

    print(f"==> Voice ID        : {voice_id}")
    print(f"==> Source WAV      : {src_wav}")
    print(f"==> Staging dataset : {RVC_DATASETS / voice_id}")
    print(f"==> Pretrained G    : {pretrained_g}")
    print(f"==> Epochs (-e)     : {args.epochs}")
    print(f"==> Save every (-c) : {args.checkpoint_every}")
    print(f"==> Output models   : {RVC_MODELS_OUT}")
    print()

    # Stage the single WAV into dataset dir expected by many RVC scripts
    staged = stage_wav(src_wav, voice_id)
    print(f"Staged: {staged}")

    # Kick off RVC training
    run_rvc_train(voice_id=voice_id,
                  total_epochs=args.epochs,
                  save_every=args.checkpoint_every,
                  pretrained_g=pretrained_g)

    # Collect and copy checkpoints into ./rvc/models/
    collect_checkpoints(voice_id)


if __name__ == "__main__":
    main()
