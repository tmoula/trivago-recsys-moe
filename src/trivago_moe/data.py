"""Dataset preprocessing for Trivago RecSys 2019 and Yoochoose (RecSys 2015).

These functions reproduce the notebook cells that produced the published
checkpoints. The id mappings depend on row order, so they are always built
from the dataframe *after* sorting by (session_id, timestamp); building them
from the unsorted CSV gives a different index space that does not match the
trained weights.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CLICKOUT = "clickout item"
TRIVAGO_COLUMNS = ["session_id", "timestamp", "action_type", "reference"]


# ---------------------------------------------------------------------------
# Trivago
# ---------------------------------------------------------------------------
def load_trivago(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv, usecols=TRIVAGO_COLUMNS)
    df = df.sort_values(["session_id", "timestamp"])
    df["action_type"] = df["action_type"].fillna("unknown")
    df["reference"] = df["reference"].fillna("0").astype(str)
    return df


def build_mappings(df: pd.DataFrame) -> dict:
    """Token ids start at 1; 0 is padding. ``df`` must come from ``load_trivago``."""
    action2idx = {a: i + 1 for i, a in enumerate(df["action_type"].unique())}
    ref2idx = {r: i + 1 for i, r in enumerate(df["reference"].unique())}
    return {
        "action2idx": action2idx,
        "ref2idx": ref2idx,
        "idx2ref": {v: k for k, v in ref2idx.items()},
        "NUM_REFS": len(ref2idx),
        "NUM_ACTION_TYPES": len(action2idx),
    }


def build_trivago_matrices(df: pd.DataFrame, mappings: dict, max_seq_len: int = 20,
                           exclude_target: bool = False):
    """One row per session that has a clickout; target = the last clicked-out hotel.

    exclude_target=False reproduces the published checkpoint: the input window is the
    last ``max_seq_len`` events of the session, which *includes* the target clickout
    itself (see "Known issues" in the README). exclude_target=True cuts the history
    just before that clickout.

    Returns (X_ref, X_action, y) as int32 arrays of shape [N, S], [N, S], [N].
    """
    df = df.assign(
        action_id=df["action_type"].map(mappings["action2idx"]),
        ref_id=df["reference"].map(mappings["ref2idx"]),
    )
    targets = (df[df["action_type"] == CLICKOUT]
               .groupby("session_id")["ref_id"].last().rename("target_ref_id"))
    sessions = df.groupby("session_id").agg(action_id=("action_id", list), ref_id=("ref_id", list))
    sessions = sessions.join(targets, how="inner")
    sessions = sessions[sessions["action_id"].apply(len) >= 2]

    clickout_id = mappings["action2idx"][CLICKOUT]
    n = len(sessions)
    X_ref = np.zeros((n, max_seq_len), dtype=np.int32)
    X_action = np.zeros((n, max_seq_len), dtype=np.int32)
    y = sessions["target_ref_id"].to_numpy(dtype=np.int32)
    keep = np.ones(n, dtype=bool)

    for i, (actions, refs) in enumerate(zip(sessions["action_id"], sessions["ref_id"])):
        if exclude_target:
            cut = len(actions) - 1 - actions[::-1].index(clickout_id)
            actions, refs = actions[:cut], refs[:cut]
        actions, refs = actions[-max_seq_len:], refs[-max_seq_len:]
        if not actions:   # clickout was the first event: nothing to predict from
            keep[i] = False
            continue
        X_action[i, max_seq_len - len(actions):] = actions
        X_ref[i, max_seq_len - len(refs):] = refs
    return X_ref[keep], X_action[keep], y[keep]


def load_item_metadata(metadata_csv: Path) -> dict:
    """hotel id (int) -> list of '|'-separated property strings."""
    meta = pd.read_csv(metadata_csv)
    return {int(k): v.split("|") for k, v in zip(meta["item_id"], meta["properties"])}


def save_mappings(mappings: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mappings, f)


def load_mappings(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Yoochoose
# ---------------------------------------------------------------------------
def load_yoochoose_sequences(clicks_dat: Path) -> tuple[list[list[int]], int]:
    """Returns (per-session item sequences with dense ids starting at 1, num_items)."""
    df = pd.read_csv(clicks_dat, names=["session_id", "timestamp", "item_id", "category"],
                     dtype={"session_id": int, "item_id": int, "category": str})
    df["clean_item_id"] = df["item_id"].astype("category").cat.codes + 1
    num_items = int(df["clean_item_id"].max())
    df = df.sort_values(["session_id", "timestamp"])
    sequences = df.groupby("session_id")["clean_item_id"].apply(list).tolist()
    return sequences, num_items


def build_next_item_matrices(sequences: list[list[int]], max_seq_len: int = 20):
    """Left-padded history (all but last item) -> last item, for sessions of length >= 2."""
    seqs = [s for s in sequences if len(s) >= 2]
    X = np.zeros((len(seqs), max_seq_len), dtype=np.int32)
    y = np.zeros(len(seqs), dtype=np.int32)
    for i, seq in enumerate(seqs):
        y[i] = seq[-1]
        hist = seq[:-1][-max_seq_len:]
        X[i, max_seq_len - len(hist):] = hist
    return X, y
