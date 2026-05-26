#!/usr/bin/env python
# -*- coding: utf-8 -*-
# RVC CPU wrapper: preprocess -> f0 -> features -> filelist.txt -> train -> collect outputs

import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "third_party" / "RVC-WebUI"
PY = sys.executable

def run(cmd, cwd=None, env=None):
    print(f'\n>>> {cmd}\n')
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)

def try_run(cmds, cwd=None, env=None):
    last = None
    for i, cmd in enumerate(cmds, 1):
        print(f'[try {i}/{len(cmds)}] {cmd}\n')
        try:
            run(cmd, cwd=cwd, env=env)
            print(f'[ok] variant {i} succeeded.')
            return
        except subprocess.CalledProcessError as e:
            print(f'[warn] variant {i} failed ({getattr(e, "returncode", "err")}).')
            last = e
    raise last

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_if_missing(src: Path, dst: Path):
    if not dst.exists() and src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)

def write_config_if_missing(exp_name: str, sr_tag: str, exp_dir: Path):
    cfg = exp_dir / "config.json"
    if cfg.exists():
        return
    # Minimal config sufficient for upstream train.py
    sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
    hop = {32000:320, 40000:400, 48000:480}[sr_map[sr_tag]]
    payload = {
        "train": {
            "log_interval": 200, "seed": 1234, "epochs": 20000,
            "learning_rate": 1e-4, "betas": [0.8, 0.99], "eps": 1e-9,
            "batch_size": 1, "fp16_run": False, "lr_decay": 0.999875,
            "segment_size": 12800, "init_lr_ratio": 1, "warmup_epochs": 0,
            "c_mel": 45, "c_kl": 1.0
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": sr_map[sr_tag],
            "filter_length": 2048,
            "hop_length": hop,
            "win_length": 2048,
            "n_mel_channels": 125,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "training_files": f"./logs/{exp_name}/filelist.txt"
        },
        "model": {
            "inter_channels": 192, "hidden_channels": 192,
            "filter_channels": 768, "n_heads": 2, "n_layers": 6,
            "kernel_size": 3, "p_dropout": 0, "resblock": "1",
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
            "upsample_rates": [10,10,2,2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16,16,4,4],
            "use_spectral_norm": False,
            "gin_channels": 256, "spk_embed_dim": 109
        },
        "model_dir": f"./logs/{exp_name}",
        "experiment_dir": f"./logs/{exp_name}",
        "save_every_epoch": 5,
        "name": exp_name,
        "total_epoch": 100,
        "pretrainG": "pretrained_v2/f0G40k.pth",
        "pretrainD": "pretrained_v2/f0D40k.pth",
        "version": "v2",
        "gpus": "0",
        "sample_rate": sr_tag,
        "if_f0": 1,
        "if_latest": 1,
        "save_every_weights": "1",
        "if_cache_data_in_gpu": 0
    }
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote default config.json -> {cfg}")

def build_filelist(exp_dir: Path, version: str = "v2", spk_id: int = 0):
    """Create logs/<exp>/filelist.txt from produced chunks/features."""
    gt = exp_dir / "0_gt_wavs"
    feat = exp_dir / ("3_feature768" if version == "v2" else "3_feature256")
    f0a = exp_dir / "2a_f0"
    f0b = exp_dir / "2b-f0nsf"
    out = exp_dir / "filelist.txt"

    if not gt.exists():
        raise FileNotFoundError(f"Missing folder: {gt}")
    # If pitch not present yet, training with f0 will fail anyway.
    missing = [p for p in [feat, f0a, f0b] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing folders: {', '.join(map(str, missing))}")

    lines = []
    for wav in sorted(gt.glob("*.wav")):
        stem = wav.stem
        line = "|".join([
            str(wav),
            str(feat / f"{stem}.npy"),
            str(f0a / f"{stem}.wav.npy"),
            str(f0b / f"{stem}.wav.npy"),
            str(spk_id),
        ])
        parts = line.split("|")
        if all(Path(p).exists() for p in parts[:-1]):
            lines.append(line)

    if not lines:
        raise RuntimeError("No matching (wav, feature, f0, f0nsf) sets found to write filelist.txt")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[info] wrote filelist with {len(lines)} entries -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--voice_id", required=True, help="Path to user_x.wav")
    ap.add_argument("-e", "--epochs", type=int, required=True)
    ap.add_argument("-c", "--chkpts", type=int, required=True)
    ap.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    print("\n===== train.py :: v2.4 (filelist builder, no datasets/) =====")
    voice_wav = Path(args.voice_id).resolve()
    if not voice_wav.exists():
        raise FileNotFoundError(f"voice file not found: {voice_wav}")
    if voice_wav.suffix.lower() != ".wav":
        print("[warn] expected a .wav file.")

    if not UPSTREAM.exists():
        raise RuntimeError(
            f"Upstream not found at {UPSTREAM}. Clone:\n"
            f'  git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI "{UPSTREAM}"'
        )

    exp_name = voice_wav.stem
    exp_dir = UPSTREAM / "logs" / exp_name
    input_dir = exp_dir / "_input"

    print(f"Using voice wav: {voice_wav}")
    print(f"Exp dir:        {exp_dir}")
    print(f"Input scratch:  {input_dir}")

    # Prepare input scratch
    if input_dir.exists():
        shutil.rmtree(input_dir, ignore_errors=True)
    ensure_dir(input_dir)
    shutil.copy2(voice_wav, input_dir / voice_wav.name)

    # Ensure assets exist in upstream tree
    copy_if_missing(PROJECT_ROOT / "rvc" / "assets" / "hubert" / "hubert_base.pt",
                    UPSTREAM / "assets" / "hubert" / "hubert_base.pt")
    copy_if_missing(PROJECT_ROOT / "rvc" / "assets" / "rmvpe" / "rmvpe.pt",
                    UPSTREAM / "assets" / "rmvpe" / "rmvpe.pt")
    copy_if_missing(PROJECT_ROOT / "rvc" / "assets" / "pretrained_v2" / "f0G40k.pth",
                    UPSTREAM / "assets" / "pretrained_v2" / "f0G40k.pth")
    copy_if_missing(PROJECT_ROOT / "rvc" / "assets" / "pretrained_v2" / "f0D40k.pth",
                    UPSTREAM / "assets" / "pretrained_v2" / "f0D40k.pth")

    # Environment for Windows/CPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["USE_LIBUV"] = "0"  # avoid "libuv not built" error on Windows
    env.setdefault("OMP_NUM_THREADS", str(max(1, args.workers)))
    env.setdefault("MKL_NUM_THREADS", str(max(1, args.workers)))
    # fairseq safe-load changed defaults in newer torch; make it permissive
    env.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

    # 1) Preprocess (multiple positional signatures exist across commits)
    sr_int = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]
    preprocess_variants = [
        # old keyword style (sometimes present)
        f'"{PY}" infer/modules/train/preprocess.py --inp_root "{input_dir}" --sr {sr_int} --n_p {args.workers} --exp_dir "{exp_dir}" --per 3.7',
        # pure positional: inp_root sr n_p exp_dir 0 3.7
        f'"{PY}" infer/modules/train/preprocess.py "{input_dir}" {sr_int} {args.workers} "{exp_dir}" 0 3.7',
        # sometimes per is last
        f'"{PY}" infer/modules/train/preprocess.py "{input_dir}" {sr_int} {args.workers} "{exp_dir}" 3.7 0',
    ]
    try_run(preprocess_variants, cwd=UPSTREAM, env=env)

    # 2) F0 (rmvpe on CPU paths fine)
    f0_variants = [
        f'"{PY}" infer/modules/train/extract/extract_f0_print.py "{exp_dir}" {args.workers} rmvpe',
        f'"{PY}" infer/modules/train/extract/extract_f0_print.py "{exp_dir}" rmvpe {args.workers}',
    ]
    try_run(f0_variants, cwd=UPSTREAM, env=env)

    # 3) HuBERT features (use stable "cpu n_part i_part exp_dir version if_vocoder" signature)
    try_run([
        f'"{PY}" infer/modules/train/extract_feature_print.py cpu {max(1,args.workers)} 0 "{exp_dir}" v2 False'
    ], cwd=UPSTREAM, env=env)

    # 4) Build filelist.txt (this was the missing piece)
    build_filelist(exp_dir, version="v2", spk_id=0)

    # 5) Ensure a minimal config (if missing)
    write_config_if_missing(exp_name, args.sr, exp_dir)

    # 6) Train
    run(
        f'"{PY}" infer/modules/train/train.py '
        f'-e "{exp_name}" -sr {args.sr} -f0 1 -bs {args.batch_size} -g 0 '
        f'-te {args.epochs} -se {args.chkpts} '
        f'-pg assets/pretrained_v2/f0G{args.sr}.pth -pd assets/pretrained_v2/f0D{args.sr}.pth '
        f'-l 0 -c 0 -sw 1 -v v2',
        cwd=UPSTREAM, env=env
    )

    # 7) Collect outputs
    models_root = ensure_dir(PROJECT_ROOT / "models")
    weights_dir = exp_dir / "weights"
    if weights_dir.exists():
        smalls = sorted(weights_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if smalls:
            shutil.copy2(smalls[-1], models_root / f"{exp_name}.pth")
    out_ckpt_dir = ensure_dir(models_root / exp_name)
    for g in sorted(exp_dir.glob("G_*.pth")):
        m = re.search(r"G_(\d+)\.pth", g.name)
        dst_name = f"e_{m.group(1)}.pth" if m else g.name
        shutil.copy2(g, out_ckpt_dir / dst_name)

    print(f"Checkpoints directory -> {out_ckpt_dir}")
    print("\nDone.")

if __name__ == "__main__":
    main()
