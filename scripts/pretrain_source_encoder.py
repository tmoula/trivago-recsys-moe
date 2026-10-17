"""Phase 1: pre-train the SASRec source encoder on Yoochoose next-click prediction
(notebook cells 16-22). Needs ~50 GB RAM and a GPU for the full 9.2M sessions.

    scripts/download_data.sh yoochoose
    python scripts/pretrain_source_encoder.py --epochs 20
"""
import argparse
import time

import torch
import torch.nn as nn

from trivago_moe import data, paths
from trivago_moe.models import SASRecSourceEncoder, TrainableSASRecSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-sessions", type=int, default=10_000,
                    help="hold out the last N sessions for HR@10")
    ap.add_argument("--out", default=str(paths.SOURCE_ENCODER_PT))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences, num_items = data.load_yoochoose_sequences(paths.YOOCHOOSE_CLICKS_DAT)
    X, y = data.build_next_item_matrices(sequences)
    del sequences
    X_tr, y_tr = X[:-args.eval_sessions], y[:-args.eval_sessions]
    X_ev, y_ev = X[-args.eval_sessions:], y[-args.eval_sessions:]
    print(f"items={num_items:,} train={len(y_tr):,} eval={len(y_ev):,}")

    model = TrainableSASRecSource(SASRecSourceEncoder(num_items)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        t0, total, batches = time.time(), 0.0, 0
        model.train()
        for i in range(0, len(y_tr), args.batch_size):
            xb = torch.from_numpy(X_tr[i:i + args.batch_size]).long().to(device)
            yb = torch.from_numpy(y_tr[i:i + args.batch_size]).long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            hits = 0
            for i in range(0, len(y_ev), args.batch_size):
                logits = model(torch.from_numpy(X_ev[i:i + args.batch_size]).long().to(device))
                yb = torch.from_numpy(y_ev[i:i + args.batch_size]).long().to(device)
                hits += (torch.topk(logits, 10).indices == yb.unsqueeze(1)).any(1).sum().item()
        print(f"epoch {epoch}/{args.epochs} loss={total / batches:.4f} "
              f"HR@10={hits / len(y_ev):.4f} ({time.time() - t0:.0f}s)")
        torch.save(model.encoder.state_dict(), args.out)


if __name__ == "__main__":
    main()
