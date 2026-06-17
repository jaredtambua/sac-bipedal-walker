import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_logs(hardcore=False):
    mode = "sac_hardcore" if hardcore else "sac_normal"

    log_dir = Path("logs") / mode
    figure_dir = Path("docs") / "figures" / mode
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = log_dir / "training_log.csv"
    eval_log_path = log_dir / "evaluation_log.csv"
    figure_path = figure_dir / "training_curve_from_logs.png"

    train_df = pd.read_csv(train_log_path)
    eval_df = pd.read_csv(eval_log_path)

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_df["episode"],
        train_df["return"],
        label="Training return",
        alpha=0.3,
    )

    plt.plot(
        train_df["episode"],
        train_df["mean_return_last_10"],
        label="10-episode training average",
    )

    if len(eval_df) > 0:
        plt.errorbar(
            eval_df["episode"],
            eval_df["mean_return"],
            yerr=eval_df["std_return"],
            fmt="o-",
            label="Evaluation mean ± std",
            capsize=4,
        )

    plt.axhline(y=300, linestyle="--", label="Solved threshold")

    title_mode = "Hardcore" if hardcore else "Normal"
    plt.title(f"SAC {title_mode} BipedalWalker Training Curve")
    plt.xlabel("Training episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {figure_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardcore", action="store_true")
    args = parser.parse_args()

    plot_logs(hardcore=args.hardcore)


if __name__ == "__main__":
    main()
