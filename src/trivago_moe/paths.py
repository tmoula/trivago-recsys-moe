"""Where data and artifacts live.

Nothing large is stored in git. Two directories are resolved at runtime:

* ``DATA_DIR``      raw public datasets (Trivago train.csv, Yoochoose clicks, ...)
* ``ARTIFACTS_DIR`` derived files and trained weights (checkpoints, mappings, P vectors)

Override with the ``TRIVAGO_MOE_DATA_DIR`` / ``TRIVAGO_MOE_ARTIFACTS_DIR`` env vars,
e.g. point both at ``/content/drive/MyDrive/Polanyi Labs`` on Colab.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("TRIVAGO_MOE_DATA_DIR", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TRIVAGO_MOE_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))

# Raw datasets
TRIVAGO_TRAIN_CSV = DATA_DIR / "train.csv"
TRIVAGO_ITEM_METADATA_CSV = DATA_DIR / "item_metadata.csv"
YOOCHOOSE_CLICKS_DAT = DATA_DIR / "yoochoose-clicks.dat"
PERSONA_EMBEDDINGS_NPY = DATA_DIR / "persona_embeddings.npy"   # 50k Nemotron personas, BGE-large
TRAVELERS_MATCHED_CSV = DATA_DIR / "travelers_matched.csv"

# Artifacts
ARCHETYPES_JSON = ARTIFACTS_DIR / "archetypes.json"
ARCHETYPE_EMBEDDINGS_NPY = ARTIFACTS_DIR / "archetype_embeddings_v2.npy"
PERSONA_P_VECTORS_NPY = ARTIFACTS_DIR / "persona_P_vectors.npy"
CONTEXT_SASREC_PT = ARTIFACTS_DIR / "context_sasrec_best.pt"
SOURCE_ENCODER_PT = ARTIFACTS_DIR / "sasrec_source_encoder.pt"
MAPPINGS_PKL = ARTIFACTS_DIR / "mappings.pkl"
