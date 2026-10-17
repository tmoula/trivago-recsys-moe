"""Maintainer script: rebuild mappings.pkl and push artifacts to the Hugging Face Hub.

Run where the data and checkpoints already are (e.g. Colab with Drive mounted):

    export TRIVAGO_MOE_DATA_DIR="/content/drive/MyDrive/Polanyi Labs"
    export TRIVAGO_MOE_ARTIFACTS_DIR="/content/drive/MyDrive/Polanyi Labs"
    huggingface-cli login
    python scripts/publish_artifacts.py

mappings.pkl is always rebuilt from the *sorted* train.csv so that it matches the
index space context_sasrec_best.pt was trained on.
"""
import argparse

from trivago_moe import data, hub, paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=hub.HF_REPO_ID)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-mappings", action="store_true")
    args = ap.parse_args()

    if not args.skip_mappings:
        print(f"Rebuilding mappings from {paths.TRIVAGO_TRAIN_CSV} ...")
        mappings = data.build_mappings(data.load_trivago(paths.TRIVAGO_TRAIN_CSV))
        assert mappings["NUM_REFS"] == 400_277 and mappings["NUM_ACTION_TYPES"] == 10, (
            "mapping sizes differ from the trained checkpoint; is this the same train.csv?")
        data.save_mappings(mappings, paths.MAPPINGS_PKL)
        print(f"✅ {paths.MAPPINGS_PKL}")

    hub.upload(repo_id=args.repo_id, private=args.private)
    print(f"✅ https://huggingface.co/{args.repo_id}")
