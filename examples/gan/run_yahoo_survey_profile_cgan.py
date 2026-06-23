import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


SURVEY_COLS = [
    "rate_frequency",
    "rate_hate",
    "rate_dislike",
    "rate_neutral",
    "rate_like",
    "rate_love",
    "preference_sensitive",
]

CONDITION_COLS = [
    "log_train_num_ratings",
    "train_avg_rating",
    "train_rating_std",
    "train_high_rating_ratio",
    "train_hate_ratio",
    "train_dislike_ratio",
    "train_neutral_ratio",
    "train_like_ratio",
    "train_love_ratio",
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
        "--out_dir",
        default=os.path.join("visual_results", "yahoo_survey_gan_profile"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-user", type=int, default=1)
    return parser.parse_args()


def load_data(metrics_path):
    df = pd.read_csv(metrics_path)
    required = ["user_id", "has_survey", "primary_group"] + SURVEY_COLS + CONDITION_COLS[1:]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns. Run examples/analysis/analyze_yahoo_survey.py first. "
            f"Missing: {missing}"
        )
    df = df[(df["has_survey"] == 1) & df["primary_group"].isin(GROUP_ORDER)].copy()
    df["log_train_num_ratings"] = np.log1p(df["train_num_ratings"].astype(float))
    df = df.dropna(subset=SURVEY_COLS + CONDITION_COLS)

    cond_mean = df[CONDITION_COLS].mean()
    cond_std = df[CONDITION_COLS].std().replace(0, 1)
    cond = ((df[CONDITION_COLS] - cond_mean) / cond_std).to_numpy(dtype=np.float32)

    target = df[SURVEY_COLS].astype(float).copy()
    for col in SURVEY_COLS[:-1]:
        target[col] = (target[col] - 1.0) / 4.0
    target["preference_sensitive"] = target["preference_sensitive"].clip(0, 1)
    return df, cond, target.to_numpy(dtype=np.float32)


def train_cgan(real_df, condition, target, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cond = torch.tensor(condition, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(cond, y), batch_size=args.batch_size, shuffle=True)

    cond_dim = condition.shape[1]
    out_dim = target.shape[1]

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(args.noise_dim + cond_dim, args.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(args.hidden_dim, out_dim),
                nn.Sigmoid(),
            )

        def forward(self, z, c):
            return self.net(torch.cat([z, c], dim=1))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(out_dim + cond_dim, args.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(args.hidden_dim, 1),
            )

        def forward(self, profile, c):
            return self.net(torch.cat([profile, c], dim=1))

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(args.epochs):
        for batch_cond, real_profile in loader:
            batch = real_profile.shape[0]
            real_label = torch.ones(batch, 1, device=device)
            fake_label = torch.zeros(batch, 1, device=device)

            z = torch.randn(batch, args.noise_dim, device=device)
            fake_profile = generator(z, batch_cond).detach()

            opt_d.zero_grad()
            loss_d = loss_fn(discriminator(real_profile, batch_cond), real_label)
            loss_d = loss_d + loss_fn(discriminator(fake_profile, batch_cond), fake_label)
            loss_d.backward()
            opt_d.step()

            z = torch.randn(batch, args.noise_dim, device=device)
            opt_g.zero_grad()
            generated = generator(z, batch_cond)
            loss_g = loss_fn(discriminator(generated, batch_cond), real_label)
            loss_g.backward()
            opt_g.step()

    generator.eval()
    with torch.no_grad():
        repeated_cond = cond.repeat_interleave(args.samples_per_user, dim=0)
        z = torch.randn(repeated_cond.shape[0], args.noise_dim, device=device)
        generated = generator(z, repeated_cond).cpu().numpy()

    user_rows = real_df.loc[real_df.index.repeat(args.samples_per_user), ["user_id", "primary_group"]]
    out = pd.DataFrame(generated, columns=SURVEY_COLS)
    out.insert(0, "primary_group", user_rows["primary_group"].to_numpy())
    out.insert(0, "user_id", user_rows["user_id"].to_numpy(dtype=int))
    out["source"] = "survey_profile_cgan"
    return denormalize_profile(out)


def denormalize_profile(df):
    out = df.copy()
    for col in SURVEY_COLS[:-1]:
        out[col] = out[col] * 4.0 + 1.0
        out[f"{col}_rounded"] = out[col].round().clip(1, 5).astype(int)
    out["preference_sensitive"] = out["preference_sensitive"].clip(0, 1)
    out["preference_sensitive_rounded"] = (out["preference_sensitive"] >= 0.5).astype(int)
    return out


def evaluate(real_df, generated_df):
    rows = []
    real_eval = real_df.copy()
    gen_eval = generated_df.copy()
    for col in SURVEY_COLS:
        for group in GROUP_ORDER:
            real_values = real_eval.loc[real_eval["primary_group"] == group, col].to_numpy(dtype=float)
            gen_values = gen_eval.loc[gen_eval["primary_group"] == group, col].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "metric": col,
                    "real_mean": real_values.mean(),
                    "generated_mean": gen_values.mean(),
                    "abs_mean_error": abs(real_values.mean() - gen_values.mean()),
                    "wasserstein": wasserstein_distance(real_values, gen_values),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df, condition, target = load_data(args.metrics)
    generated = train_cgan(real_df, condition, target, args)
    metrics = evaluate(real_df, generated)
    summary = (
        metrics.groupby("metric")[["abs_mean_error", "wasserstein"]]
        .mean()
        .reset_index()
    )

    generated_path = out_dir / "generated_survey_profile.csv"
    metrics_path = out_dir / "distribution_metrics.csv"
    summary_path = out_dir / "summary_by_metric.csv"
    generated.to_csv(generated_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"saved: {generated_path}")
    print(f"saved: {metrics_path}")
    print(f"saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
