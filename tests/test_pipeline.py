"""Fast CPU smoke tests on synthetic data: no datasets or weights needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trivago_moe import data, personas
from trivago_moe.models import (ContextAwareSASRec, FullCrossDomainPipeline, MultiTaskCrossDomainLoss,
                                PersonaConditionedMoEBridge, SASRecSourceEncoder, TrainableSASRecSource)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def tiny_trivago(tmp_path):
    rows = [
        # session, ts, action, reference
        ("s2", 1, "search for destination", "Paris"),
        ("s2", 2, "interaction item image", "444"),
        ("s2", 3, "clickout item", "444"),
        ("s1", 1, "clickout item", "4"),          # only clickout is the first event:
        ("s1", 2, "change of sort order", None),  # dropped when excluding the target
        ("s3", 1, "filter selection", "Pool"),    # no clickout: never a sample
    ]
    p = tmp_path / "train.csv"
    pd.DataFrame(rows, columns=data.TRIVAGO_COLUMNS).to_csv(p, index=False)
    return p


def test_trivago_preprocessing(tmp_path):
    df = data.load_trivago(tiny_trivago(tmp_path))
    m = data.build_mappings(df)
    # ids follow first appearance in (session_id, timestamp) order: s1 comes first
    assert m["action2idx"]["clickout item"] == 1
    assert m["ref2idx"]["4"] == 1 and m["ref2idx"]["0"] == 2
    assert m["idx2ref"][m["ref2idx"]["444"]] == "444"

    Xr, Xa, y = data.build_trivago_matrices(df, m, max_seq_len=4)
    assert Xr.shape == (2, 4) and list(y) == [m["ref2idx"]["4"], m["ref2idx"]["444"]]
    assert Xr[1, -1] == y[1]  # published setup: target clickout is inside the history

    assert (Xa[1] > 0).sum() == 3

    Xr, Xa, y = data.build_trivago_matrices(df, m, max_seq_len=4, exclude_target=True)
    assert list(y) == [m["ref2idx"]["444"]]
    assert (Xa[0] > 0).sum() == 2 and Xa[0, -1] == m["action2idx"]["interaction item image"]


def test_next_item_matrices():
    X, y = data.build_next_item_matrices([[1], [1, 2, 3], list(range(1, 30))], max_seq_len=5)
    assert X.shape == (2, 5)
    assert list(X[0]) == [0, 0, 0, 1, 2] and y[0] == 3
    assert list(X[1]) == [24, 25, 26, 27, 28] and y[1] == 29


def test_model_shapes_and_loss():
    B, S, D = 4, 20, 32
    enc = SASRecSourceEncoder(num_items=50, hidden_dim=D, num_heads=2, num_layers=1)
    clicks = torch.randint(1, 51, (B, S))
    assert enc(clicks).shape == (B, D)
    assert TrainableSASRecSource(enc)(clicks).shape == (B, 51)

    bridge = PersonaConditionedMoEBridge(D, D, 7)
    P = torch.softmax(torch.randn(B, 7), -1)
    pipe = FullCrossDomainPipeline(bridge, num_travel_items=30, hidden_dim=D)
    logits, H_t_hat, H_t = pipe(enc(clicks), P, torch.randint(1, 31, (B, S)))
    assert logits.shape == (B, 31) and H_t_hat.shape == H_t.shape == (B, D)
    loss = MultiTaskCrossDomainLoss()(logits, torch.randint(1, 31, (B,)), H_t_hat, P)
    loss.backward()
    assert bridge.experts.grad is not None


def test_checkpoint_keys_match_notebook():
    keys = set(ContextAwareSASRec(10, 3).state_dict())
    assert {"ref_embedding.weight", "action_embedding.weight", "position_embedding.weight",
            "transformer.layers.1.linear2.weight", "layer_norm.weight", "head.weight"} <= keys
    enc_keys = set(SASRecSourceEncoder(10).state_dict())
    assert "transformer_blocks.layers.3.self_attn.in_proj_weight" in enc_keys


def test_recommend_end_to_end(tmp_path):
    from recommend import recommend

    df = data.load_trivago(tiny_trivago(tmp_path))
    m = data.build_mappings(df)
    model = ContextAwareSASRec(m["NUM_REFS"], m["NUM_ACTION_TYPES"], hidden_dim=16).eval()
    recs = recommend(model, m, [("clickout item", "4"), ("unknown action", "999")], k=3)
    assert len(recs) == 3 and all(h in m["ref2idx"] for h, _ in recs)


def test_persona_vectors():
    rng = np.random.default_rng(0)
    archetypes = {str(i): {"label": f"A{i}", "centroids": dict(zip(personas.CENTROID_FEATURES,
                                                                   rng.random(6)))} for i in range(7)}
    travelers = pd.DataFrame({"archetype_label": ["A0", "A3", "A6"], "match_score": [.5, .4, .45]})
    P = personas.behavioral_p_vectors(archetypes, travelers)
    assert P.shape == (3, 7) and np.allclose(P.sum(1), 1)
    assert P[1].argmax() == 3

    P2 = personas.sharpened_p_vectors(rng.random((5, 16)), rng.random((7, 16)), top_k=3)
    assert np.allclose(P2.sum(1), 1) and ((P2 > 0).sum(1) == 3).all()
