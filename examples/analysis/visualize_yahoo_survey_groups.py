import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PRIMARY_ORDER = [
    "daily + preference_affects",
    "daily + not_affects",
    "less_than_daily + preference_affects",
    "less_than_daily + not_affects",
]

POLARITY_ORDER = ["negative", "neutral", "positive"]
EXTREME_ORDER = ["extreme_selective", "other"]


def load_group_frames(out_dir: str, stem: str):
    summary = pd.read_csv(os.path.join(out_dir, f"{stem}_summary.csv"))
    counts = pd.read_csv(os.path.join(out_dir, f"{stem}_counts.csv"))
    return summary, counts


def reorder(df: pd.DataFrame, key: str, order: list[str]) -> pd.DataFrame:
    order_map = {name: idx for idx, name in enumerate(order)}
    if key in df.columns:
        df = df.copy()
        df["_order"] = df[key].map(order_map).fillna(len(order_map))
        df = df.sort_values("_order").drop(columns="_order")
    return df


def draw_bar(ax, labels, values, title, ylabel, color):
    ax.bar(range(len(labels)), values, color=color, width=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def draw_group_panel(ax_row, summary, counts, group_col, title_prefix):
    labels = summary[group_col].tolist()
    draw_bar(ax_row[0], labels, counts["user_count"].tolist(), f"{title_prefix}: user count", "users", "#4C78A8")
    draw_bar(ax_row[1], labels, summary["train_avg_rating"].tolist(), f"{title_prefix}: train avg rating", "rating", "#59A14F")
    draw_bar(ax_row[2], labels, summary["test_avg_rating"].tolist(), f"{title_prefix}: test avg rating", "rating", "#E15759")


def make_overview_figure(out_dir: str):
    primary_summary, primary_counts = load_group_frames(out_dir, "primary_group")
    polarity_summary, polarity_counts = load_group_frames(out_dir, "polarity_group")
    extreme_summary, extreme_counts = load_group_frames(out_dir, "extreme_group")

    primary_summary = reorder(primary_summary, "primary_group", PRIMARY_ORDER)
    primary_counts = reorder(primary_counts, "primary_group", PRIMARY_ORDER)
    polarity_summary = reorder(polarity_summary, "polarity_group", POLARITY_ORDER)
    polarity_counts = reorder(polarity_counts, "polarity_group", POLARITY_ORDER)
    extreme_summary = reorder(extreme_summary, "extreme_group", EXTREME_ORDER)
    extreme_counts = reorder(extreme_counts, "extreme_group", EXTREME_ORDER)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    draw_group_panel(axes[0], primary_summary, primary_counts, "primary_group", "Primary")
    draw_group_panel(axes[1], polarity_summary, polarity_counts, "polarity_group", "Polarity")
    draw_group_panel(axes[2], extreme_summary, extreme_counts, "extreme_group", "Extreme")

    fig.suptitle("Yahoo survey group overview", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "group_overview_new.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_primary_detail_figure(out_dir: str):
    summary, counts = load_group_frames(out_dir, "primary_group")
    summary = reorder(summary, "primary_group", PRIMARY_ORDER)
    counts = reorder(counts, "primary_group", PRIMARY_ORDER)
    labels = summary["primary_group"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    draw_bar(axes[0, 0], labels, counts["user_count"].tolist(), "Primary: user count", "users", "#4C78A8")
    draw_bar(axes[0, 1], labels, summary["train_num_ratings"].tolist(), "Primary: train num ratings", "ratings", "#76B7B2")
    draw_bar(axes[0, 2], labels, summary["train_avg_rating"].tolist(), "Primary: train avg rating", "rating", "#59A14F")
    draw_bar(axes[1, 0], labels, summary["train_high_rating_ratio"].tolist(), "Primary: train high-rating ratio", "ratio", "#F28E2B")
    draw_bar(axes[1, 1], labels, summary["test_avg_rating"].tolist(), "Primary: test avg rating", "rating", "#E15759")
    draw_bar(axes[1, 2], labels, summary["test_high_rating_ratio"].tolist(), "Primary: test high-rating ratio", "ratio", "#B07AA1")

    fig.suptitle("Yahoo survey primary grouping detail", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "primary_group_detail.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("visual_results", "yahoo_survey"),
    )
    args = parser.parse_args()

    overview_path = make_overview_figure(args.out_dir)
    detail_path = make_primary_detail_figure(args.out_dir)
    print("Saved overview figure:", overview_path)
    print("Saved primary detail figure:", detail_path)


if __name__ == "__main__":
    main()
