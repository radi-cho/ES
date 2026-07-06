"""Exit 1 if validation.csv indicates a broken run (flat ~0 scores)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def validation_is_bad(csv_path: Path) -> str | None:
    if not csv_path.is_file() or csv_path.stat().st_size <= len("epoch,validation_score,time_seconds\n"):
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "validation_score" not in df.columns:
        return None
    post_warmup = df[df["epoch"] >= 5] if (df["epoch"] >= 5).any() else df
    if post_warmup.empty:
        return None
    scores = post_warmup["validation_score"].astype(float)
    if scores.max() <= 0.015:
        return f"max validation {scores.max():.4f} <= 0.015 after epoch 5"
    if (scores <= 0.015).any():
        bad = post_warmup.loc[scores <= 0.015, ["epoch", "validation_score"]]
        row = bad.iloc[0]
        return f"validation {row['validation_score']:.4f} at epoch {int(row['epoch'])} <= 0.015"
    return None


def main() -> None:
    path = Path(sys.argv[1])
    reason = validation_is_bad(path)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
