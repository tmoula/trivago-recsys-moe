# Contributing

## Setup

```bash
pip install -e ".[dev]"          # add ,personas for sentence-transformers/faiss/scikit-learn
pytest
```

The tests run on synthetic data in a few seconds, so you can run them without downloading anything.

## Ground rules

- **Never commit data or weights.** `.gitignore` blocks the usual extensions (`*.pt`, `*.npy`, `*.csv`, ...). If you produce a new artifact, add its filename to `PUBLISHED_FILES` in `src/trivago_moe/hub.py` and ask a maintainer to publish it with `scripts/publish_artifacts.py`.
- **Keep model classes state-dict compatible.** The published checkpoints load by parameter name. If you rename a submodule or change a shape, that needs a new checkpoint and a note in the README.
- **Paths go through `trivago_moe.paths`.** Never hardcode `/content/drive/...`.
- **New logic belongs in `src/`, not notebooks.** Notebooks are for exploration. Clear their outputs before committing (`nbstripout notebooks/your_notebook.ipynb`), except for `polanyilabs_prototype.ipynb`, whose outputs are the record of the original runs.
- **Report metrics with their split.** Say whether a number is train, val or test, and which preprocessing produced it (for example `--exclude-target`).

## Good first contributions

- A script version of the Phase 2/3 MoE training in notebook §3.
- Leak-free baseline numbers (`train_context_sasrec.py --exclude-target`), added to the README.
- A real retail-to-travel user alignment for the MoE stage, instead of index-modulo pairing.
- Scripts that rebuild `archetypes.json` and `persona_embeddings.npy`.

## Pull requests

Keep them focused, include `pytest` output, and for any modelling change include the before/after metrics and the command that produced them.
