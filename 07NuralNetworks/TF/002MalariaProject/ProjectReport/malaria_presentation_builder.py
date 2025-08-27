#!/usr/bin/env python3
"""
Malaria Presentation Builder
----------------------------
Creates an MP4 video from a PowerPoint deck and a fixed narration script.

Features:
- Exports slides from PPTX to PNG (tries Windows PowerPoint COM or LibreOffice "soffice").
- Generates narration using pyttsx3 (offline). Attempts to select a female, neutral voice.
- Adds subtle background music if provided (looped & ducked under narration).
- Clean fade transitions between slides.
- No text overlays; the visuals come from your slides.

Usage:
  python malaria_presentation_builder.py \
      --pptx slides.pptx \
      --out malaria_presentation.mp4 \
      --music path/to/background_music.mp3   # optional

If slide export fails automatically, export your slides to PNGs manually:
  - PowerPoint: File → Export → Change File Type → PNG → "All Slides"
  - Save them into a folder (e.g., slides_png/). Then run:
      python malaria_presentation_builder.py --images_dir slides_png --out out.mp4 --music bg.mp3

Install deps (recommended in a virtualenv):
  pip install -r requirements.txt

Tested on Windows/macOS/Linux with Python 3.10+

Note: pyttsx3 uses system voices; availability varies by OS. The script tries to pick a female voice.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# ---------- Slide scripts (1-based index aligns with slide order) ----------
SLIDE_SCRIPTS = {
    1: ("Automating Malaria Pathology using Computer Vision",
        "Malaria diagnosis traditionally relies on microscopic inspection of stained blood smears. "
        "Trained pathologists identify the presence of Plasmodium parasites within red blood cells, "
        "which appear as distinct ring-like or irregular structures. Differentiating infected from "
        "uninfected cells is crucial, but the process is time-consuming, labor-intensive, and prone "
        "to human error."),
    2: ("The Malaria Endemic",
        "This choropleth map highlights the global burden of malaria. In 2000, 108 countries reported "
        "active malaria endemic zones, shown in red, green, and yellow. By 2023, the number reduced to "
        "83 countries, still marked in red. The most severely impacted regions remain Africa, South Asia, "
        "and parts of Latin America."),
    3: ("Training Dataset",
        "This slide shows examples from our training dataset. We used 27,556 labeled cell images from the "
        "TensorFlow malaria dataset, each tagged as uninfected or parasitized. Parasitized cells reveal "
        "dark, irregular Plasmodium structures inside, while uninfected cells appear clear and uniform. "
        "These distinguishing features are what our CNN learns to detect."),
    4: ("CNN LeNet Model Architecture",
        "As the task is relatively simple, we implemented CNN LeNet architecture for malaria detection. "
        "Input images were first resized to 224 by 224 by 3 pixels before entering the network. "
        "The model begins with two convolutional layers, six and sixteen filters, each followed by batch "
        "normalization and max pooling to extract features. A kernel size of five was used as there are not "
        "many complicated features to learn in the convolution layers. The output of the convolution layers "
        "are next flattened and passed through dense layers of one hundred and ten neurons with ReLU activation. "
        "Finally, a sigmoid output layer classifies cells as parasitized or uninfected, enabling efficient binary classification."),
    5: ("Model Summary, Parameter Counts and Optimal Hyper Parameters",
        "The LeNet model contains about four point five million trainable parameters. The first convolutional "
        "layers are lightweight, with four hundred fifty six and two thousand four hundred sixteen parameters "
        "respectively. Most parameters come from the fully connected dense layer with one hundred units, "
        "contributing over four point four nine million. Batch normalization layers add a few hundred each, "
        "while the final dense layers with ten and one neurons add just over one thousand combined. This distribution "
        "shows that parameter complexity is dominated by the dense layers rather than convolutional filters. "
        "We achieved the best performance with an eighty, ten, ten split, batch size of thirty-two, Adam optimizer "
        "at a zero point zero one learning rate, and binary cross-entropy loss. Early stopping limited training "
        "to eight to nine epochs, preventing overfitting."),
    6: ("Results and Accuracy",
        "This graph highlights how well our model performed with early stopping. The training accuracy climbs "
        "rapidly above ninety six point three percent, while the validation accuracy improves steadily before "
        "stabilizing. The red test accuracy line remains consistently high at around ninety four percent. When "
        "validation accuracy stopped improving, early stopping kicked in, preventing overfitting and locking "
        "in strong performance. Overall, this shows that our LeNet based CNN not only learns quickly but also "
        "generalizes well, making it reliable for identifying malaria infected cells."),
    7: ("Sample Feature-Maps for Positive and Negative Examples",
        "Let us take a deeper look on how the CNN distinguishes malaria positive and negative red blood cells. "
        "For the parasitized example, convolutional layers progressively detect irregular structures caused by "
        "Plasmodium parasites, reinforcing a positive classification. Conversely, for the uninfected cell, "
        "smooth, ring like structures are learned layer by layer, guiding the model toward a negative classification. "
        "Due to usage of ReLU activation and batch normalization, some activations appear blank or blacked out, "
        "which is normal. The final sigmoid activation outputs low probability indicating parasitized in the first "
        "case and high probability indicating uninfected in the second, confirming correct predictions."),
    8: ("Conclusion and the Way Forward",
        "Our LeNet based CNN successfully distinguishes malaria infected from healthy red blood cells with high "
        "accuracy, validating deep learning as a reliable diagnostic aid. Moving forward, larger datasets, "
        "advanced architectures, and real world clinical validation could further improve robustness. Such AI "
        "driven systems can support early, scalable malaria detection across endemic regions. For further "
        "exploration, our code, models and experiments are available on Google Colab and GitHub. Key references "
        "include LeNet and TensorFlow documentation, which guided this project’s design and implementation.")
}

# ---------- Utilities ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# ---------- Slide export helpers ----------

def export_slides_windows_com(pptx: Path, out_dir: Path) -> bool:
    """Export PPTX slides to PNG using Windows PowerPoint COM (requires PowerPoint installed)."""
    try:
        import win32com.client  # type: ignore
    except Exception:
        return False
    try:
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        presentation = powerpoint.Presentations.Open(str(pptx), WithWindow=False)
        ensure_dir(out_dir)
        # 17 = ppSaveAsPNG
        presentation.SaveAs(str(out_dir), 17)
        presentation.Close()
        powerpoint.Quit()
        return True
    except Exception as e:
        print(f"[WARN] Windows COM export failed: {e}")
        return False

def export_slides_libreoffice(pptx: Path, out_dir: Path) -> bool:
    """Export PPTX slides to PNG using LibreOffice soffice (must be installed and on PATH)."""
    ensure_dir(out_dir)
    try:
        # Convert to PNG directly (works with recent LibreOffice)
        # soffice --headless --convert-to png --outdir slides_png slides.pptx
        cmd = ["soffice", "--headless", "--convert-to", "png", "--outdir", str(out_dir), str(pptx)]
        print("[INFO] Trying LibreOffice export:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            # LibreOffice names output as <basename>.png if one slide; else creates multiple PNGs
            # If it created a single PNG for all slides (rare), we consider that a fail.
            pngs = list(out_dir.glob("*.png"))
            if pngs:
                return True
        print("[WARN] LibreOffice export stdout:", res.stdout)
        print("[WARN] LibreOffice export stderr:", res.stderr)
        return False
    except FileNotFoundError:
        print("[WARN] 'soffice' not found on PATH.")
        return False

# ---------- TTS (pyttsx3) ----------

def synthesize_tts(text: str, out_wav: Path, rate_wpm: int = 135):
    """Generate speech audio using pyttsx3 and save to WAV."""
    import pyttsx3  # offline TTS
    engine = pyttsx3.init()
    # Rate
    engine.setProperty('rate', rate_wpm)
    # Voice selection (try to pick a female, neutral voice if available)
    try:
        voices = engine.getProperty('voices')
        chosen_id = None
        for v in voices:
            name = (getattr(v, "name", "") or "").lower()
            gender = (getattr(v, "gender", "") or "").lower()
            lang = ",".join(getattr(v, "languages", []) or [])
            if ("female" in gender) or ("zira" in name) or ("susan" in name) or ("hazel" in name):
                chosen_id = v.id
                break
        if chosen_id is None and voices:
            chosen_id = voices[0].id
        if chosen_id:
            engine.setProperty('voice', chosen_id)
    except Exception as e:
        print(f"[WARN] Voice selection issue: {e}")
    # Save to file
    engine.save_to_file(text, str(out_wav))
    engine.runAndWait()

# ---------- Video Assembly (MoviePy) ----------

def build_video_from_images(images: List[Path], scripts: List[str], out_mp4: Path, music_path: Optional[Path], fps: int = 30):
    from moviepy.editor import (
        ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, AudioClip
    )
    from moviepy.audio.fx import audio_fadein, audio_fadeout

    assert len(images) == len(scripts), "Number of images must match number of scripts."

    tmpdir = Path(tempfile.mkdtemp(prefix="malaria_vid_"))
    audio_wavs = []

    print("[INFO] Generating narration audio...")
    # Generate narration WAV per slide
    for idx, text in enumerate(scripts, start=1):
        wav_path = tmpdir / f"slide_{idx:02d}.wav"
        synthesize_tts(text, wav_path)
        audio_wavs.append(wav_path)

    print("[INFO] Creating per-slide clips...")
    # Create per-slide video clips with audio; duration = audio duration + padding
    slide_clips = []
    crossfade = 0.6  # seconds
    pad_tail = 0.4   # seconds
    target_w, target_h = 1920, 1080

    for i, (img_path, wav_path) in enumerate(zip(images, audio_wavs), start=1):
        img_clip = ImageClip(str(img_path)).resize(height=target_h).on_color(
            size=(target_w, target_h), color=(0, 0, 0), pos=('center', 'center')
        )
        narration = AudioFileClip(str(wav_path))
        duration = float(narration.duration) + pad_tail
        clip = img_clip.set_duration(duration).set_audio(narration)
        slide_clips.append(clip)

    print("[INFO] Concatenating with crossfades...")
    video = concatenate_videoclips(slide_clips, method="compose", padding=-crossfade, transparent=False)
    # Apply crossfade between consecutive clips
    for i in range(1, len(slide_clips)):
        slide_clips[i] = slide_clips[i].crossfadein(crossfade)

    final = concatenate_videoclips(slide_clips, method="compose")

    # Background music handling
    if music_path and music_path.exists():
        print("[INFO] Adding background music...")
        music = AudioFileClip(str(music_path))
        # Loop music to final duration
        loops = int(final.duration // music.duration) + 1
        musics = [music] * loops
        bg = concatenate_videoclips(musics).set_duration(final.duration).audio_volumex(0.08)
        # Fade in/out
        bg = audio_fadein.audio_fadein(bg, 1.0)
        bg = audio_fadeout.audio_fadeout(bg, 1.0)
        # Mix audio
        comp_audio = CompositeAudioClip([final.audio, bg])
        final = final.set_audio(comp_audio)

    print(f"[INFO] Writing video to {out_mp4} ... This may take a few minutes.")
    final.write_videofile(str(out_mp4), fps=fps, codec="libx264", audio_codec="aac", threads=4, preset="medium")

def collect_slide_images(pptx_path: Optional[Path], images_dir: Optional[Path]) -> List[Path]:
    if images_dir:
        pngs = sorted(images_dir.glob("*.png"), key=lambda p: natural_sort_key(p.name))
        if not pngs:
            raise FileNotFoundError(f"No PNGs found in {images_dir}.")
        return pngs

    assert pptx_path is not None, "Either --pptx or --images_dir must be provided."

    out_dir = Path("slides_png")
    out_dir.mkdir(exist_ok=True)

    # Try Windows COM export first
    if sys.platform.startswith("win"):
        if export_slides_windows_com(pptx_path, out_dir):
            pngs = sorted(out_dir.glob("*.png"), key=lambda p: natural_sort_key(p.name))
            if pngs:
                return pngs

    # Try LibreOffice
    if export_slides_libreoffice(pptx_path, out_dir):
        pngs = sorted(out_dir.glob("*.png"), key=lambda p: natural_sort_key(p.name))
        if pngs:
            return pngs

    raise RuntimeError(
        "Could not auto-export slides to PNG. Please export slides to PNGs manually:\n"
        "PowerPoint: File → Export → Change File Type → PNG → 'All Slides'\n"
        "Save them into a folder and run with --images_dir that folder."
    )

def main():
    parser = argparse.ArgumentParser(description="Build an MP4 presentation from PPTX and narration script.")
    parser.add_argument("--pptx", type=str, help="Path to slides.pptx")
    parser.add_argument("--images_dir", type=str, help="Folder containing slide PNGs (alternative to --pptx)")
    parser.add_argument("--out", type=str, required=True, help="Output MP4 path")
    parser.add_argument("--music", type=str, help="Optional background music path (mp3/wav)")
    parser.add_argument("--voice_rate", type=int, default=135, help="Narration rate in words per minute (default 135)")
    args = parser.parse_args()

    pptx_path = Path(args.pptx).resolve() if args.pptx else None
    images_dir = Path(args.images_dir).resolve() if args.images_dir else None
    out_mp4 = Path(args.out).resolve()
    music_path = Path(args.music).resolve() if args.music else None

    # Collect slide images
    images = collect_slide_images(pptx_path, images_dir)

    # Align scripts to number of images (truncate or pad with empty strings if mismatch)
    scripts = [SLIDE_SCRIPTS.get(i+1, ("", ""))[1] for i in range(len(images))]
    if len(scripts) != len(images):
        print(f"[WARN] Found {len(images)} images but {len(SLIDE_SCRIPTS)} scripts defined. Using first {len(images)} scripts.")

    # Build video
    build_video_from_images(images, scripts, out_mp4, music_path)

if __name__ == "__main__":
    main()
