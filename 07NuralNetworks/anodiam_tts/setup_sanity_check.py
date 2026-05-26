import torch
import torchaudio
import faiss
import librosa
import soundfile

print("=== Python Dependencies Sanity Check ===")

# Torch + CUDA
print("torch:", torch.__version__, "| CUDA available?", torch.cuda.is_available())

# Torchaudio
print("torchaudio:", torchaudio.__version__)

# Faiss
print("faiss GPU support?", hasattr(faiss, "StandardGpuResources"))

# Librosa
print("librosa:", librosa.__version__)

# Soundfile
print("soundfile:", soundfile.__libsndfile_version__)
