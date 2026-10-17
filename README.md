# trivago-recsys-moe

Session-based hotel recommendation on the [Trivago RecSys Challenge 2019](https://recsys.acm.org/recsys19/challenge/) data ([Kaggle mirror](https://www.kaggle.com/datasets/pranavmahajan725/trivagorecsyschallengedata2019)), with an experimental **persona-conditioned mixture-of-experts (MoE) bridge** that transfers retail browsing behaviour ([Yoochoose, RecSys Challenge 2015](https://recsys.acm.org/recsys15/challenge/); [Kaggle mirror](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015)) into the travel domain.

Built as the Polanyi Labs prototype. The original Colab notebook is in [`notebooks/`](notebooks/polanyilabs_prototype.ipynb). Its logic is factored into an installable package (`src/trivago_moe`) and runnable scripts.

## Architecture

```
            Yoochoose clicks (9.2M sessions)        Nemotron personas (50k) + 7 behavioural archetypes
                        │                                            │
  Phase 1   SASRecSourceEncoder  ──►  H_s [256]         persona mixture  P [7]
                        │                                            │
  Phase 2               └──────►  PersonaConditionedMoEBridge  ◄─────┘
                                   gate = softmax(MLP(P));  Ĥ_t = Σ_e gate_e · W_e · H_s
                                              │
  Phase 3                       FullCrossDomainPipeline  ──►  hotel logits (363,874 hotels)

  Baseline  ContextAwareSASRec: Trivago (hotel, action-type) sequence ──► next clickout hotel
```

| Component | Params | Trained on | Reported result |
|---|---|---|---|
| `SASRecSourceEncoder` (+ head) | ~30M | Yoochoose, 7.99M sessions, 20 epochs | train loss 3.58; HR@10 0.53 *(on training sessions)* |
| `PersonaConditionedMoEBridge` + `FullCrossDomainPipeline` | ~188M | Trivago, 610k sessions, 20 epochs | joint loss 5.47 (no held-out eval) |
| `ContextAwareSASRec` | 103M | Trivago, 649k sessions (80/10/10 split) | test HR@10 0.753, HR@20 0.755 ⚠️ see [Known issues](#known-issues) |

## Repository layout

```
src/trivago_moe/
  models.py       all model classes (state-dict compatible with the notebook checkpoints)
  data.py         Trivago / Yoochoose preprocessing and id mappings
  personas.py     persona mixture (P vector) construction
  hub.py          download / upload artifacts on the Hugging Face Hub
  paths.py        data + artifact locations (env-var configurable)
scripts/
  download_data.sh            public datasets from Kaggle
  download_weights.py         trained artifacts from the Hugging Face Hub
  recommend.py                top-k hotels for a session
  train_context_sasrec.py     train + evaluate ContextAwareSASRec
  pretrain_source_encoder.py  Phase 1 Yoochoose pre-training
  build_persona_vectors.py    persona_P_vectors.npy
  publish_artifacts.py        (maintainers) rebuild mappings + push weights to the Hub
notebooks/polanyilabs_prototype.ipynb   original exploratory notebook, with outputs
tests/                                   CPU smoke tests on synthetic data
```

## Quickstart

```bash
git clone https://github.com/tmoula/trivago-recsys-moe && cd trivago-recsys-moe
pip install -e ".[dev]"
pytest                                   # ~3 s, no data needed

python scripts/download_weights.py context_sasrec_best.pt mappings.pkl
python scripts/recommend.py --events "search for item:0" "clickout item:4" "clickout item:444"
```

Add `--metadata data/item_metadata.csv` (after `scripts/download_data.sh`) to print stars, rating and amenities for each hotel.

## Data and weights: why they are not in this repo

GitHub rejects files over 100 MB, and Git LFS only gives 1 GB of free storage and bandwidth. This project needs several GB, so **git holds code only**. Large files live in two places:

| File | Size | Where it comes from |
|---|---|---|
| `train.csv`, `test.csv`, `item_metadata.csv` | ~780 MB zipped | [Kaggle](https://www.kaggle.com/datasets/pranavmahajan725/trivagorecsyschallengedata2019): `scripts/download_data.sh trivago` |
| `yoochoose-clicks.dat` | ~1.5 GB | [Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015): `scripts/download_data.sh yoochoose` |
| `context_sasrec_best.pt` | ~413 MB | Hugging Face Hub: `scripts/download_weights.py` |
| `sasrec_source_encoder.pt` | ~67 MB | Hugging Face Hub |
| `mappings.pkl`, `archetypes.json`, `archetype_embeddings_v2.npy`, `persona_P_vectors.npy` | < 50 MB each | Hugging Face Hub |
| `hs_vectors.npy` | ~8.2 GB | not hosted; regenerate from the source encoder (notebook §2) |
| `X_*_matrix.npy`, `y_*.npy` | ~50 MB each | not hosted; rebuilt on the fly by `trivago_moe.data` |

By default, datasets go in `./data` and artifacts in `./artifacts`, and both are gitignored. You can point them anywhere:

```bash
export TRIVAGO_MOE_DATA_DIR=/path/to/data
export TRIVAGO_MOE_ARTIFACTS_DIR=/path/to/artifacts
export TRIVAGO_MOE_HF_REPO=tmoula/trivago-recsys-moe   # Hub repo with the weights
```

**On Colab**, set both directories to `/content/drive/MyDrive/<folder>` after mounting Drive. The notebook's first cell does this for you.

## Reproducing training

| Step | Command | Hardware used |
|---|---|---|
| Trivago baseline | `python scripts/train_context_sasrec.py` | 1×A100, ~8 min (20 epochs) |
| Leak-free variant | `python scripts/train_context_sasrec.py --exclude-target` | same |
| Phase 1 pre-training | `python scripts/pretrain_source_encoder.py` | 1×A100, ~50 GB RAM, ~5.5 min/epoch |
| Persona mixtures | `python scripts/build_persona_vectors.py` | CPU |

Phase 2/3 (MoE bridge) training still lives only in notebook §3. A script version is an open contribution.

## Known issues

These come from reading the prototype. Please read them before you build on the numbers above.

1. **Target leakage in the Trivago baseline.** The input window covers the session's last 20 events, and that window *includes the target clickout itself*. The model can learn to copy the last reference, which is why HR@10 (0.753) and HR@20 (0.755) are almost identical. `--exclude-target` cuts the history before the target. Expect much lower but honest numbers.
2. **The published `mappings.pkl` may not match the checkpoint.** The notebook trained on ids built from the CSV *sorted* by `(session_id, timestamp)`, but exported `mappings.pkl` from the *unsorted* CSV. `pandas.unique()` follows row order, so the ids differ, and predictions would decode to the wrong hotels. `scripts/publish_artifacts.py` rebuilds the mappings the same way as training and checks their sizes (400,277 refs, 10 actions).
3. **The MoE stage pairs unrelated rows.** In Phase 2/3, retail vectors, personas and travel sessions are joined by `idx % len(...)`. No real user links them, so the bridge is trained on arbitrary pairings. Its logits also come from `Ĥ_t` alone, and the travel-side transformer output `H_t` goes unused.
4. **Phase 1 HR@10 is measured on training sessions.** The notebook evaluates on the last 10k sessions, which were part of training. `pretrain_source_encoder.py` holds them out.
5. **Upstream persona inputs are not reproducible from this repo.** `archetypes.json` (K-Means over Yoochoose), `persona_embeddings.npy` (BGE-large over Nemotron personas) and `travelers_matched.csv` were built outside the notebook.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The Known issues above are good first contributions.

## License

Code is MIT licensed (see [LICENSE](LICENSE)). The datasets keep their own terms: the Trivago RecSys 2019 and Yoochoose data are released for research use, so check their licenses before any commercial use.
