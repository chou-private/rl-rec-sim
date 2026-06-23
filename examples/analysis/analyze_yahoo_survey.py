import argparse
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("YAHOO_DISABLE_GAN_PROFILE", "1")

sys.path.extend([".", "./src", "./src/DeepCTR-Torch"])

from src.core.envs.YahooR3.YahooData import YahooData


USER_TYPE_MAP = {
    0: "no_survey",
    1: "silent",
    2: "active",
    3: "extreme",
    4: "positive",
    5: "negative",
    6: "mixed",
}


def compute_user_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grp = df.groupby("user_id")["rating"]
    df_user_metrics = grp.agg(
        num_ratings="count",
        avg_rating="mean",
        rating_std="std",
    ).reset_index()
    df_user_metrics["high_rating_ratio"] = grp.apply(lambda x: (x >= 4).mean()).to_numpy()
    rating_dist = (
        df.pivot_table(
            index="user_id",
            columns="rating",
            values="item_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
        .astype(float)
    )
    rating_dist = rating_dist.div(rating_dist.sum(axis=1), axis=0).fillna(0)
    rating_dist.columns = [
        "hate_ratio",
        "dislike_ratio",
        "neutral_ratio",
        "like_ratio",
        "love_ratio",
    ]
    df_user_metrics = df_user_metrics.merge(
        rating_dist.reset_index(), on="user_id", how="left"
    )
    df_user_metrics["rating_std"] = df_user_metrics["rating_std"].fillna(0)
    rename_map = {
        col: f"{prefix}_{col}"
        for col in [
            "num_ratings",
            "avg_rating",
            "rating_std",
            "high_rating_ratio",
            "hate_ratio",
            "dislike_ratio",
            "neutral_ratio",
            "like_ratio",
            "love_ratio",
        ]
    }
    df_user_metrics = df_user_metrics.rename(columns=rename_map)
    return df_user_metrics


def add_analysis_groups(df_user_all: pd.DataFrame) -> pd.DataFrame:
    df_user_all = df_user_all.copy()

    df_user_all["activity_group"] = np.where(
        df_user_all["rate_frequency"] == 5, "daily", "less_than_daily"
    )
    df_user_all["sensitivity_group"] = np.where(
        df_user_all["preference_sensitive"] == 1,
        "preference_affects",
        "not_affects",
    )
    df_user_all["primary_group"] = (
        df_user_all["activity_group"] + " + " + df_user_all["sensitivity_group"]
    )

    positivity_bias_raw = (
        (df_user_all["rate_like"] + df_user_all["rate_love"]) / 2
        - (df_user_all["rate_hate"] + df_user_all["rate_dislike"]) / 2
    )
    df_user_all["polarity_group"] = np.where(
        positivity_bias_raw > 0,
        "positive",
        np.where(positivity_bias_raw < 0, "negative", "neutral"),
    )

    extreme_selective_mask = (
        ((df_user_all["rate_hate"] + df_user_all["rate_love"]) / 2 >= 5)
        & (df_user_all["rate_neutral"] <= 3)
    )
    df_user_all["extreme_group"] = np.where(
        extreme_selective_mask, "extreme_selective", "other"
    )

    return df_user_all


def summarize_by_group(df_user_all: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    summary = df_user_all.groupby(group_col)[metrics].mean()
    summary["user_count"] = df_user_all.groupby(group_col)["user_id"].count()
    summary = summary.reset_index()
    return summary


def compute_stats(df_user_all: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    try:
        from scipy import stats
    except ImportError:
        print("scipy is not available; skip statistical tests.")
        return pd.DataFrame()

    results = []
    groups = [g for g in df_user_all[group_col].unique() if pd.notna(g)]

    for metric in metrics:
        data_by_group = [
            df_user_all.loc[df_user_all[group_col] == g, metric].dropna().to_numpy()
            for g in groups
        ]
        if len(data_by_group) < 2:
            continue
        try:
            f_stat, p_val = stats.f_oneway(*data_by_group)
        except Exception:
            f_stat, p_val = np.nan, np.nan
        results.append(
            {"metric": metric, "test": "anova", "group_a": "all", "group_b": "all", "stat": f_stat, "p_value": p_val}
        )

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a = data_by_group[i]
                b = data_by_group[j]
                if len(a) < 2 or len(b) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                results.append(
                    {
                        "metric": metric,
                        "test": "welch_t",
                        "group_a": groups[i],
                        "group_b": groups[j],
                        "stat": t_stat,
                        "p_value": p_val,
                    }
                )

    return pd.DataFrame(results)


def save_group_outputs(
    df_user_all: pd.DataFrame,
    group_col: str,
    metrics: list[str],
    out_dir: str,
    stem: str,
) -> None:
    summary = summarize_by_group(df_user_all, group_col, metrics)
    summary_path = os.path.join(out_dir, f"{stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved group summary ({stem}):", summary_path)
    print(f"\nGroup summary ({stem}):")
    print(summary.to_string(index=False))

    stats_df = compute_stats(df_user_all, group_col, metrics)
    if not stats_df.empty:
        stats_path = os.path.join(out_dir, f"{stem}_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved stats ({stem}):", stats_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("visual_results", "yahoo_survey"),
    )
    args = parser.parse_args()

    dataset = YahooData()
    df_train, df_user, _, _ = dataset.get_train_data()
    df_test, _, _, _ = dataset.get_val_data()

    df_train_metrics = compute_user_metrics(df_train, prefix="train")
    df_test_metrics = compute_user_metrics(df_test, prefix="test")

    df_user_all = df_train_metrics.merge(
        df_test_metrics, on="user_id", how="outer"
    ).merge(
        df_user.reset_index(), on="user_id", how="left"
    )
    df_user_all["has_survey"] = df_user_all["has_survey"].fillna(0).astype(int)
    df_user_all["user_type"] = df_user_all["user_type"].fillna(0).astype(int)
    df_user_all["user_type_label"] = df_user_all["user_type"].map(USER_TYPE_MAP)
    df_user_all = add_analysis_groups(df_user_all)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "user_metrics.csv")

    df_user_all.to_csv(metrics_path, index=False)

    print("Saved user-level metrics:", metrics_path)

    df_survey = df_user_all[df_user_all["has_survey"] == 1].copy()
    all_metrics = [
        "train_num_ratings",
        "train_avg_rating",
        "train_rating_std",
        "train_high_rating_ratio",
        "train_hate_ratio",
        "train_dislike_ratio",
        "train_neutral_ratio",
        "train_like_ratio",
        "train_love_ratio",
        "test_num_ratings",
        "test_avg_rating",
        "test_rating_std",
        "test_high_rating_ratio",
        "test_hate_ratio",
        "test_dislike_ratio",
        "test_neutral_ratio",
        "test_like_ratio",
        "test_love_ratio",
    ]

    group_specs = [
        ("primary_group", "primary_group"),
        ("polarity_group", "polarity_group"),
        ("extreme_group", "extreme_group"),
        ("user_type_label", "legacy_user_type"),
    ]

    for group_col, stem in group_specs:
        counts_path = os.path.join(args.out_dir, f"{stem}_counts.csv")
        df_survey[group_col].value_counts().rename_axis(group_col).reset_index(
            name="user_count"
        ).to_csv(counts_path, index=False)
        print(f"Saved group counts ({stem}):", counts_path)
        save_group_outputs(
            df_user_all=df_survey,
            group_col=group_col,
            metrics=all_metrics,
            out_dir=args.out_dir,
            stem=stem,
        )


if __name__ == "__main__":
    main()
