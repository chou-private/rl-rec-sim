import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SURVEY_COLS = [
    "rate_frequency",
    "rate_hate",
    "rate_dislike",
    "rate_neutral",
    "rate_like",
    "rate_love",
    "preference_sensitive",
]

GROUP_ORDER = [
    "daily + preference_affects",
    "daily + not_affects",
    "less_than_daily + preference_affects",
    "less_than_daily + not_affects",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default=os.path.join("visual_results", "yahoo_survey", "user_metrics.csv"),
    )
    parser.add_argument(
        "--generated",
        default=os.path.join(
            "visual_results", "yahoo_survey_gan_profile", "generated_survey_profile.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("visual_results", "yahoo_survey_gan_profile"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.metrics)
    real = real[(real["has_survey"] == 1) & real["primary_group"].isin(GROUP_ORDER)]
    generated = pd.read_csv(args.generated)

    real_mean = real.groupby("primary_group")[SURVEY_COLS].mean().loc[GROUP_ORDER]
    gen_mean = generated.groupby("primary_group")[SURVEY_COLS].mean().loc[GROUP_ORDER]

    rows = []
    for group in GROUP_ORDER:
        for col in SURVEY_COLS:
            rows.append(
                {
                    "group": group,
                    "metric": col,
                    "real_mean": real_mean.loc[group, col],
                    "generated_mean": gen_mean.loc[group, col],
                    "abs_error": abs(real_mean.loc[group, col] - gen_mean.loc[group, col]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "group_mean_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes = axes.ravel()
    for ax, col in zip(axes, SURVEY_COLS):
        x = range(len(GROUP_ORDER))
        width = 0.38
        ax.bar([i - width / 2 for i in x], real_mean[col], width=width, label="real", color="#355070")
        ax.bar([i + width / 2 for i in x], gen_mean[col], width=width, label="generated", color="#e76f51")
        ax.set_title(col)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"G{i+1}" for i in x])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].axis("off")
    axes[0].legend(frameon=False)
    fig.suptitle("Real vs GAN-generated survey profile means", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "survey_profile_group_mean_comparison.png", dpi=180)
    fig.savefig(out_dir / "survey_profile_group_mean_comparison.svg")

    print(f"saved: {out_dir / 'group_mean_comparison.csv'}")
    print(f"saved: {out_dir / 'survey_profile_group_mean_comparison.png'}")
    print(f"saved: {out_dir / 'survey_profile_group_mean_comparison.svg'}")


if __name__ == "__main__":
    main()
