from huggingface_hub import hf_hub_download
import os

# ensure all directories exist
dirs = [
    "rvc/assets/hubert",
    "rvc/assets/pretrained",
    "rvc/assets/pretrained_v2",
    "rvc/assets/rmvpe",
    "rvc/assets/uvr5_weights"
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# download HuBERT
hubert = hf_hub_download(
    repo_id="AI-C/rvc-models",
    filename="hubert_base.pt",
    repo_type="dataset",
    local_dir="rvc/assets/hubert",
    local_dir_use_symlinks=False
)

# download RMVPE
rmvpe = hf_hub_download(
    repo_id="lj1995/VoiceConversionWebUI",
    filename="rmvpe.pt",
    local_dir="rvc/assets/rmvpe",
    local_dir_use_symlinks=False
)

# download pretrained D40k (example)
d40k = hf_hub_download(
    repo_id="lj1995/VoiceConversionWebUI",
    filename="pretrained/f0D40k.pth",
    local_dir="rvc/assets/pretrained",
    local_dir_use_symlinks=False
)

print("Downloaded:", hubert, rmvpe, d40k)
