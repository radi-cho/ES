"""Plot validation, train eval, and update norms for D30 metrics runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_hours"] = df["time_seconds"] / 3600.0
    for col in ("validation_score", "train_eval_score", "active_pool_train_eval_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_metrics_compare(
    curves: list[tuple[Path, str]],
    output_path: Path,
    title: str | None = None,
) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 10.5), sharex=True)

    panels = [
        ("validation_score", "Validation score (256 held-out)"),
        ("train_eval_score", "Train eval (all 30 prompts)"),
        ("active_pool_train_eval_score", "Train eval (active pool only)"),
        ("total_update_rms", "Parameter update RMS (mean per tensor)"),
    ]

    for metrics_csv, label in curves:
        df = _load_metrics(metrics_csv)
        for ax, (col, ylabel) in zip(axes, panels):
            if col not in df.columns:
                continue
            series = df.dropna(subset=[col])
            if series.empty:
                continue
            ax.plot(series["time_hours"], series[col], linewidth=2, label=label)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wall-clock time (hours)")
    if title is None:
        title = "D30 full vs phased — validation, train fit, update norm"
    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path, nargs="+")
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("-o", "--output", type=Path, default=Path("runs/D30_metrics_full_vs_phased.png"))
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    labels = args.label or [f"run {i+1}" for i in range(len(args.metrics_csv))]
    if len(labels) != len(args.metrics_csv):
        raise SystemExit("Provide one --label per metrics_csv")
    curves = list(zip(args.metrics_csv, labels))
    out = plot_metrics_compare(curves, args.output, title=args.title)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
