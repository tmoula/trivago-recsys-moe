"""Train the Trivago ContextAwareSASRec (notebook cells 40-42) and report test HR@k.

    scripts/download_data.sh trivago
    python scripts/train_context_sasrec.py                    # reproduces the published setup
    python scripts/train_context_sasrec.py --exclude-target   # leak-free history (see README)
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from trivago_moe import data, paths
from trivago_moe.models import ContextAwareSASRec


@torch.no_grad()
def hit_rate(model, Xr, Xa, y, k, batch_size, device):
    model.eval()
    hits = 0
    for i in range(0, len(y), batch_size):
        logits = model(Xr[i:i + batch_size].to(device), Xa[i:i + batch_size].to(device))
        topk = torch.topk(logits, k, dim=1).indices
        hits += (topk == y[i:i + batch_size].to(device).unsqueeze(1)).any(dim=1).sum().item()
    model.train()
    return hits / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-target", action="store_true",
                    help="cut each session's history before the target clickout")
    ap.add_argument("--out", default=str(paths.CONTEXT_SASREC_PT))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = data.load_trivago(paths.TRIVAGO_TRAIN_CSV)
    mappings = data.build_mappings(df)
    data.save_mappings(mappings, paths.MAPPINGS_PKL)
    Xr, Xa, y = (torch.from_numpy(m).long() for m in
                 data.build_trivago_matrices(df, mappings, exclude_target=args.exclude_target))
    del df
    n = len(y)
    print(f"sessions={n:,} refs={mappings['NUM_REFS']:,} actions={mappings['NUM_ACTION_TYPES']}")

    idx = torch.from_numpy(rng.permutation(n))
    tr, va, te = idx[:int(.8 * n)], idx[int(.8 * n):int(.9 * n)], idx[int(.9 * n):]

    model = ContextAwareSASRec(mappings["NUM_REFS"], mappings["NUM_ACTION_TYPES"],
                               hidden_dim=args.hidden_dim).to(device)
    print(f"parameters={sum(p.numel() for p in model.parameters()):,} device={device}")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        t0, total, batches = time.time(), 0.0, 0
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xr[b].to(device), Xa[b].to(device)), y[b].to(device))
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        scheduler.step()
        val = hit_rate(model, Xr[va], Xa[va], y[va], 10, args.batch_size, device)
        print(f"epoch {epoch}/{args.epochs} loss={total / batches:.4f} val HR@10={val:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val > best:
            best = val
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device))
    for k in (10, 20):
        print(f"test HR@{k}={hit_rate(model, Xr[te], Xa[te], y[te], k, args.batch_size, device):.4f}")


if __name__ == "__main__":
    main()
