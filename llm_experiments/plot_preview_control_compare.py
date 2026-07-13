"""Plot the PR6 validation curve against the matched PR4/PR5 curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _curve(path: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    required = {"time_seconds", "validation_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame["time_seconds"] / 3600.0, frame["validation_score"]


def plot_compare(
    *,
    pr6: Path,
    baseline: Path,
    preview_ridge: Path | None,
    output: Path,
    virtual_population: int,
    budget_hours: float,
) -> None:
    plt.figure(figsize=(7.0, 4.5))
    curves: list[tuple[Path, str, str]] = [
        (pr6, f"PR6 centered+calibrated (virtual={virtual_population})", "C2"),
    ]
    if preview_ridge is not None:
        curves.append(
            (preview_ridge, f"Preview ridge (virtual={virtual_population})", "C0")
        )
    curves.append((baseline, "Baseline EGGROLL (pop=64)", "C1"))

    for path, label, color in curves:
        x, y = _curve(path)
        plt.plot(x, y, linewidth=2.2, label=label, color=color)

    plt.xlabel("Wall-clock time (hours)")
    plt.ylabel("Validation score")
    plt.title("Countdown D8 validation — prefill oracle vs baseline (Qwen3.5-2B)")
    plt.xlim(0.0, budget_hours)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr6", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--preview-ridge", type=Path)
    parser.add_argument("--output", type=Path, default=Path("preview_control_compare.png"))
    parser.add_argument("--virtual-population", type=int, default=1024)
    parser.add_argument("--budget-hours", type=float, default=2.0)
    args = parser.parse_args()
    plot_compare(
        pr6=args.pr6,
        baseline=args.baseline,
        preview_ridge=args.preview_ridge,
        output=args.output,
        virtual_population=args.virtual_population,
        budget_hours=args.budget_hours,
    )
    print(f"Saved: {args.output}")
    print(f"Saved: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
