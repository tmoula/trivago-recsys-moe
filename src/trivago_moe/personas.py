"""Persona mixture vectors P (one 7-dim distribution over archetypes per traveler).

Each archetype is one of 7 behavioural clusters (K-Means over Yoochoose sessions,
stored in archetypes.json). The 50k Nemotron personas are embedded with
BAAI/bge-large-en-v1.5, and P is a softened match of each persona to the archetypes.
"""
import numpy as np
from scipy.special import softmax

CENTROID_FEATURES = ["n_clicks", "duration_secs", "n_return_clicks",
                     "n_categories", "avg_gap_secs", "conversion_rate"]


def behavioral_p_vectors(archetypes: dict, travelers, temperature: float = 0.3) -> np.ndarray:
    """The method saved as persona_P_vectors.npy (notebook cell 7).

    ``travelers`` is the travelers_matched.csv dataframe (columns archetype_label,
    match_score). For each traveler, take the cosine similarity of its dominant
    archetype's standardized behavioural centroid to every archetype's centroid,
    scale by the BGE match score, and apply a temperature softmax.
    """
    from sklearn.preprocessing import StandardScaler

    n = len(archetypes)
    feats = np.array([[archetypes[str(i)]["centroids"][f] for f in CENTROID_FEATURES]
                      for i in range(n)])
    arch = StandardScaler().fit_transform(feats)
    arch = arch / np.linalg.norm(arch, axis=1, keepdims=True)
    centroid_cos = arch @ arch.T                                     # [7, 7]

    label_to_id = {v["label"]: int(k) for k, v in archetypes.items()}
    dominant = travelers["archetype_label"].map(label_to_id).to_numpy()
    score = travelers["match_score"].to_numpy(dtype=float)[:, None]
    return softmax(centroid_cos[dominant] * score / temperature, axis=1)


def sharpened_p_vectors(persona_emb: np.ndarray, archetype_emb: np.ndarray,
                        temperature: float = 0.15, top_k: int = 7) -> np.ndarray:
    """Alternative explored in notebook cell 10: per-row min-max on cosine
    similarity, temperature softmax, optional top-k sparsification."""
    p = persona_emb / np.linalg.norm(persona_emb, axis=1, keepdims=True)
    a = archetype_emb / np.linalg.norm(archetype_emb, axis=1, keepdims=True)
    sims = p @ a.T
    lo, hi = sims.min(axis=1, keepdims=True), sims.max(axis=1, keepdims=True)
    scaled = (sims - lo) / np.where(hi - lo == 0, 1.0, hi - lo)
    probs = softmax(scaled / temperature, axis=1)
    if top_k < probs.shape[1]:
        keep = np.argsort(probs, axis=1)[:, -top_k:]
        mask = np.zeros_like(probs)
        np.put_along_axis(mask, keep, 1.0, axis=1)
        probs = probs * mask
        probs /= probs.sum(axis=1, keepdims=True)
    return probs
