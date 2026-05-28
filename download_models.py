"""
Download all VoxFlow model weights.

Models downloaded:
  HuggingFace (ASLP-lab/MeanVC):
    - fastu2++.pt       (ASR encoder, 92MB)
    - meanvc_200ms.pt   (VC DiT, 56MB)
    - vocos.pt          (Vocos vocoder, 33MB)

  Google Drive (manual or auto via gdown):
    - wavlm_large_finetune.pth  (ECAPA-TDNN speaker verification, ~50MB)
      Required for full speaker similarity quality.
      Without it, zero speaker embeddings are used (reduced quality).
"""

import sys
from pathlib import Path

MODELS_DIR = Path("models")
HF_REPO = "ASLP-lab/MeanVC"
HF_FILES = ["fastu2++.pt", "meanvc_200ms.pt", "vocos.pt"]

# Google Drive file ID for wavlm_large_finetune.pth
# From: https://github.com/ASLP-lab/MeanVC (see their README for the link)
WAVLM_GDRIVE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"


def download_hf_models():
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MeanVC models from HuggingFace ({HF_REPO})...")

    for filename in HF_FILES:
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        print(f"  Downloading {filename}...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  {filename} done.")


def download_wavlm(gdrive_id: str = None):
    dest = MODELS_DIR / "wavlm_large_finetune.pth"
    if dest.exists():
        print("wavlm_large_finetune.pth already exists.")
        return

    if gdrive_id is None:
        print("\n[WARNING] wavlm_large_finetune.pth not downloaded.")
        print("  This file enables full speaker similarity quality.")
        print("  To get it:")
        print("  1. Check https://github.com/ASLP-lab/MeanVC for the download link")
        print("  2. Place it in: models/wavlm_large_finetune.pth")
        print("  3. Or run: python download_models.py --wavlm-id <GOOGLE_DRIVE_FILE_ID>")
        print("\n  VoxFlow will still work without it (using zero speaker embeddings).\n")
        return

    try:
        import gdown
        print(f"Downloading wavlm_large_finetune.pth from Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(dest), quiet=False)
        print("wavlm_large_finetune.pth done.")
    except ImportError:
        print("Install gdown: pip install gdown")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually and place in models/wavlm_large_finetune.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download VoxFlow model weights")
    parser.add_argument("--wavlm-id", type=str, default=WAVLM_GDRIVE_ID,
                        help="Google Drive file ID for wavlm_large_finetune.pth")
    parser.add_argument("--hf-only", action="store_true",
                        help="Only download HuggingFace models (skip wavlm)")
    args = parser.parse_args()

    download_hf_models()
    if not args.hf_only:
        download_wavlm(args.wavlm_id)

    print("\nDone. Model files in:", MODELS_DIR.resolve())
    print("Files:")
    for f in MODELS_DIR.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
