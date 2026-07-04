"""Combined EGGROLL vs GRPO validation curves (solid/dashed, D256 vs small pool)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _latest_validation_csv(repo_root: Path, pattern: str) -> Path | None:
    matches = sorted(repo_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    csv_path = matches[-1] / "validation.csv"
    return csv_path if csv_path.is_file() else None


def discover_2h_d8_d256_csvs(repo_root: Path) -> dict[str, Path]:
    specs = {
        "eggroll_d256": "countdown_chat_eggroll_D256_2h_disjoint_rand_*trainD=256*",
        "eggroll_d8": "countdown_chat_eggroll_D8_2h_disjoint_rand_*trainD=8*",
        "grpo_d256": "countdown_chat_grpo_D256_2h_disjoint_rand_*trainD=256*",
        "grpo_d8": "countdown_chat_grpo_D8_2h_disjoint_rand_*trainD=8*",
    }
    out: dict[str, Path] = {}
    for key, pattern in specs.items():
        path = _latest_validation_csv(repo_root, pattern)
        if path is not None:
            out[key] = path
    return out


def plot_eggroll_vs_grpo(
    curves: list[tuple[Path, str, str, str]],
    output_path: Path,
    title: str,
    xmax_hours: float | None = None,
) -> Path:
    plt.figure(figsize=(7.0, 4.5))
    for csv_path, label, linestyle, color in curves:
        df = pd.read_csv(csv_path)
        required = {"validation_score", "time_seconds"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        x_hours = df["time_seconds"] / 3600.0
        plt.plot(
            x_hours,
            df["validation_score"],
            linewidth=2,
            label=label,
            linestyle=linestyle,
            color=color,
        )

    plt.xlabel("Wall-clock time (hours)")
    plt.ylabel("Validation score")
    plt.title(title)
    plt.ylim(bottom=0.0)
    if xmax_hours is not None:
        plt.xlim(0.0, xmax_hours)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EGGROLL vs GRPO comparison figure")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: parent of llm_experiments)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    parser.add_argument(
        "--small-d",
        type=int,
        default=8,
        help="Small train pool size for legend (default: 8)",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=2.0,
        help="Time budget in hours (x-axis limit)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until all four validation.csv files exist",
    )
    parser.add_argument(
        "--wait-log",
        type=Path,
        default=None,
        help="If set, wait until this log contains 'All runs complete'",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=120,
        help="Poll interval when --wait is used",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out = args.output or (
        repo_root / "runs" / f"figure_eggroll_vs_grpo_disjoint_random_D{args.small_d}_vs_D256_{int(args.budget_hours)}h.png"
    )

    if args.wait_log is not None:
        import time

        wait_log = args.wait_log.resolve()
        while not wait_log.is_file() or "All runs complete" not in wait_log.read_text():
            print(f"Waiting for {wait_log.name} …")
            time.sleep(args.poll_seconds)

    if args.wait:
        import time

        required_keys = ("eggroll_d256", "eggroll_d8", "grpo_d256", "grpo_d8")
        while True:
            found = discover_2h_d8_d256_csvs(repo_root)
            missing = [k for k in required_keys if k not in found]
            if not missing:
                break
            print(f"Missing runs: {missing}; sleeping {args.poll_seconds}s")
            time.sleep(args.poll_seconds)
        csvs = found
    else:
        csvs = discover_2h_d8_d256_csvs(repo_root)
        missing = [k for k in ("eggroll_d256", "eggroll_d8", "grpo_d256", "grpo_d8") if k not in csvs]
        if missing:
            raise SystemExit(f"Missing validation.csv for: {missing}. Use --wait or rerun after experiments finish.")

    d_small = args.small_d
    curves = [
        (csvs["eggroll_d256"], "EGGROLL train D=256", "-", "C0"),
        (csvs["eggroll_d8"], f"EGGROLL train D={d_small}", "-", "C1"),
        (csvs["grpo_d256"], "GRPO train D=256", "--", "C0"),
        (csvs["grpo_d8"], f"GRPO train D={d_small}", "--", "C1"),
    ]
    title = (
        f"Countdown validation — Qwen3.5-2B (EGGROLL vs GRPO, disjoint+random 8/epoch, {args.budget_hours:g}h)"
    )
    saved = plot_eggroll_vs_grpo(curves, out, title, xmax_hours=args.budget_hours)
    print(f"Saved: {saved}")
    print(f"Saved: {saved.with_suffix('.pdf')}")
    for key, path in csvs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
