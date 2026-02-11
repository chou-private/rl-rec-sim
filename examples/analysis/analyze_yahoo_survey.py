import argparse
import os
import sys

import numpy as np
import pandas as pd

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


def compute_user_metrics(df_train: pd.DataFrame) -> pd.DataFrame:
    grp = df_train.groupby("user_id")["rating"]
    df_user_metrics = grp.agg(
        num_ratings="count",
        avg_rating="mean",
        rating_std="std",
    ).reset_index()
    df_user_metrics["high_rating_ratio"] = grp.apply(lambda x: (x >= 4).mean()).to_numpy()
    df_user_metrics["rating_std"] = df_user_metrics["rating_std"].fillna(0)
    return df_user_metrics


def summarize_by_group(df_user_all: pd.DataFrame, group_col: str) -> pd.DataFrame:
    metrics = ["num_ratings", "avg_rating", "rating_std", "high_rating_ratio"]
    summary = df_user_all.groupby(group_col)[metrics].mean()
    summary["user_count"] = df_user_all.groupby(group_col)["user_id"].count()
    summary = summary.reset_index()
    return summary


def compute_stats(df_user_all: pd.DataFrame, group_col: str) -> pd.DataFrame:
    try:
        from scipy import stats
    except ImportError:
        print("scipy is not available; skip statistical tests.")
        return pd.DataFrame()

    metrics = ["num_ratings", "avg_rating", "rating_std", "high_rating_ratio"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("visual_results", "yahoo_survey"),
    )
    parser.add_argument(
        "--only_survey",
        action="store_true",
        help="If set, only include users with survey answers (has_survey=1).",
    )
    args = parser.parse_args()

    dataset = YahooData()
    df_train, df_user, _, _ = dataset.get_train_data()

    df_user_metrics = compute_user_metrics(df_train)
    df_user_all = df_user_metrics.merge(
        df_user.reset_index(), on="user_id", how="left"
    )
    df_user_all["has_survey"] = df_user_all["has_survey"].fillna(0).astype(int)
    df_user_all["user_type"] = df_user_all["user_type"].fillna(0).astype(int)
    df_user_all["user_type_label"] = df_user_all["user_type"].map(USER_TYPE_MAP)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "user_metrics.csv")

    df_user_all.to_csv(metrics_path, index=False)

    print("Saved user-level metrics:", metrics_path)

    type_counts_path = os.path.join(args.out_dir, "user_type_counts.csv")
    df_user_all["user_type_label"].value_counts().rename_axis("user_type").reset_index(
        name="user_count"
    ).to_csv(type_counts_path, index=False)
    print("Saved user type counts:", type_counts_path)

    if args.only_survey:
        df_survey = df_user_all[df_user_all["has_survey"] == 1].copy()
        summary_survey = summarize_by_group(df_survey, "user_type_label")
        summary_path = os.path.join(args.out_dir, "group_summary_survey_only.csv")
        summary_survey.to_csv(summary_path, index=False)
        print("Saved group summary (survey only):", summary_path)
        print("\nGroup summary (survey only):")
        print(summary_survey.to_string(index=False))
        stats_df = compute_stats(df_survey, "user_type_label")
        if not stats_df.empty:
            stats_path = os.path.join(args.out_dir, "group_stats_survey_only.csv")
            stats_df.to_csv(stats_path, index=False)
            print("Saved stats (survey only):", stats_path)
        return

    summary_all = summarize_by_group(df_user_all, "user_type_label")
    summary_path_all = os.path.join(args.out_dir, "group_summary_all.csv")
    summary_all.to_csv(summary_path_all, index=False)
    print("Saved group summary (all users):", summary_path_all)

    df_survey = df_user_all[df_user_all["has_survey"] == 1].copy()
    summary_survey = summarize_by_group(df_survey, "user_type_label")
    summary_path = os.path.join(args.out_dir, "group_summary_survey_only.csv")
    summary_survey.to_csv(summary_path, index=False)
    print("Saved group summary (survey only):", summary_path)

    print("\nGroup summary (survey only):")
    print(summary_survey.to_string(index=False))

    stats_df = compute_stats(df_survey, "user_type_label")
    if not stats_df.empty:
        stats_path = os.path.join(args.out_dir, "group_stats_survey_only.csv")
        stats_df.to_csv(stats_path, index=False)
        print("Saved stats (survey only):", stats_path)


if __name__ == "__main__":
    main()
