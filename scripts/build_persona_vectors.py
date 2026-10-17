"""Build persona_P_vectors.npy ([N_travelers, 7]) from archetypes.json + travelers_matched.csv
(notebook cell 7).

    python scripts/build_persona_vectors.py --temperature 0.3
"""
import argparse
import json

import numpy as np
import pandas as pd

from trivago_moe import paths
from trivago_moe.personas import behavioral_p_vectors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    with open(paths.ARCHETYPES_JSON) as f:
        archetypes = json.load(f)
    travelers = pd.read_csv(paths.TRAVELERS_MATCHED_CSV)
    P = behavioral_p_vectors(archetypes, travelers, args.temperature)
    np.save(paths.PERSONA_P_VECTORS_NPY, P)
    print(f"✅ {paths.PERSONA_P_VECTORS_NPY} {P.shape}, mean dominant weight {P.max(1).mean():.3f}")
