"""Fetch / publish trained artifacts on the Hugging Face Hub.

Weights and derived files are too large for GitHub (100 MB per-file limit, 1 GB
free LFS quota), so they live in a Hugging Face model repo instead.
"""
import os
from pathlib import Path

from . import paths

HF_REPO_ID = os.environ.get("TRIVAGO_MOE_HF_REPO", "tmoula/trivago-recsys-moe")

# Files needed to run inference / Phase 2-3 training without the raw datasets.
PUBLISHED_FILES = [
    "context_sasrec_best.pt",      # ~413 MB  ContextAwareSASRec weights
    "mappings.pkl",                # action/ref <-> id mappings for the checkpoint above
    "sasrec_source_encoder.pt",    # Phase 1 retail encoder (Yoochoose)
    "archetypes.json",             # 7 behavioural archetypes
    "archetype_embeddings_v2.npy", # BGE embeddings of archetype descriptions
    "persona_P_vectors.npy",       # [50000, 7] persona mixtures
]


def download(filenames=None, repo_id: str = HF_REPO_ID, local_dir: Path = paths.ARTIFACTS_DIR) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames or PUBLISHED_FILES:
        hf_hub_download(repo_id=repo_id, filename=name, local_dir=local_dir)
    return local_dir


def upload(filenames=None, repo_id: str = HF_REPO_ID, local_dir: Path = paths.ARTIFACTS_DIR,
           private: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    for name in filenames or PUBLISHED_FILES:
        f = local_dir / name
        if not f.exists():
            print(f"skip (missing): {f}")
            continue
        api.upload_file(path_or_fileobj=str(f), path_in_repo=name, repo_id=repo_id)
        print(f"uploaded {name}")
