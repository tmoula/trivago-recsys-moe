"""Download trained weights and small artifacts from the Hugging Face Hub into artifacts/.

    python scripts/download_weights.py                 # everything
    python scripts/download_weights.py mappings.pkl    # specific files
"""
import sys

from trivago_moe import hub

if __name__ == "__main__":
    out = hub.download(sys.argv[1:] or None)
    print(f"✅ artifacts in {out}")
