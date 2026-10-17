"""Top-k hotel recommendations for a session, using the trained ContextAwareSASRec.

    python scripts/download_weights.py context_sasrec_best.pt mappings.pkl
    python scripts/recommend.py \
        --events "search for item:0" "filter selection:0" "clickout item:4" "clickout item:444" "change of sort order:0"

Each event is "<action_type>:<reference>". Pass --metadata data/item_metadata.csv
to print hotel properties alongside ids.
"""
import argparse

import torch

from trivago_moe import data, paths
from trivago_moe.models import ContextAwareSASRec

KEY_AMENITIES = {"WiFi (Rooms)", "Free WiFi (Rooms)", "Pool", "Gym", "Restaurant",
                 "Spa (Wellness Facility)", "Pet Friendly", "Family Friendly", "Airport Shuttle"}


def load_model(device, checkpoint=paths.CONTEXT_SASREC_PT, mappings_path=paths.MAPPINGS_PKL):
    mappings = data.load_mappings(mappings_path)
    model = ContextAwareSASRec(num_refs=mappings["NUM_REFS"], num_actions=mappings["NUM_ACTION_TYPES"])
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval(), mappings


def recommend(model, mappings, events, k=10, max_seq_len=20, device="cpu"):
    """events: list of (action_type, reference) -> list of (original hotel id, score)."""
    events = events[-max_seq_len:]
    actions = [mappings["action2idx"].get(a, 0) for a, _ in events]
    refs = [mappings["ref2idx"].get(str(r), 0) for _, r in events]
    pad = max_seq_len - len(events)
    a = torch.tensor([[0] * pad + actions], device=device)
    r = torch.tensor([[0] * pad + refs], device=device)
    with torch.no_grad():
        scores = torch.softmax(model(r, a), dim=-1)[0]
    scores[0] = 0  # padding token
    top = torch.topk(scores, k)
    return [(mappings["idx2ref"].get(i, "?"), s) for i, s in zip(top.indices.tolist(), top.values.tolist())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True, help='"<action_type>:<reference>" ...')
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--metadata", default=None, help="path to item_metadata.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mappings = load_model(device)
    events = [tuple(e.rsplit(":", 1)) for e in args.events]
    meta = data.load_item_metadata(args.metadata) if args.metadata else {}

    print(f"Top {args.k} hotels:")
    for rank, (hotel_id, score) in enumerate(recommend(model, mappings, events, args.k, device=device), 1):
        line = f"{rank:>3}. hotel {hotel_id:<10} p={score:.4f}"
        props = meta.get(int(hotel_id)) if str(hotel_id).isdigit() else None
        if props:
            stars = next((p for p in props if "Star" in p and "From" not in p), "?")
            rating = next((p for p in props if "Rating" in p), "?")
            amen = [p for p in props if p in KEY_AMENITIES][:3]
            line += f"  {stars} | {rating} | {', '.join(amen) or '-'}"
        print(line)
