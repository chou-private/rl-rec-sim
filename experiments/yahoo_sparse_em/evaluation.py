"""Random-song rating metrics and matched comparisons for the sparse-history experiment."""
import hashlib

import numpy as np

from experiments.yahoo_em.run import bootstrap_difference


ARMS = ("global", "own", "shuffled")
CONTRASTS = (("own", "global"), ("own", "shuffled"), ("shuffled", "global"))
METRICS = ("mae", "rmse", "nll", "ndcg5")


def evaluate_rows(rows, probabilities, *, original_user_ids=None):
    """Score aligned N×5 probabilities on zero-indexed (user, song, star) rows.

    Each user has ten random songs in Yahoo R3. ID is a deterministic tie
    breaker for ranking; binary relevance is a four- or five-star rating.
    """
    rows = np.asarray(rows)
    p = np.asarray(probabilities, dtype=np.float64)
    if (rows.ndim != 2 or rows.shape[1] != 3 or len(rows) == 0
            or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0)
            or np.any(rows[:, 2] >= 5)):
        raise ValueError("Expected nonempty zero-based (user, song, star) rows")
    if (p.shape != (len(rows), 5) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Invalid five-star probability distributions")
    if original_user_ids is not None:
        original_user_ids = np.asarray(original_user_ids)
        if original_user_ids.ndim != 1 or rows[:, 0].max() >= len(original_user_ids):
            raise ValueError("No original ID mapping for some evaluation users")

    order = np.lexsort((rows[:, 1], rows[:, 0]))
    rows, p = rows[order], p[order]
    if np.any(np.all(rows[1:, :2] == rows[:-1, :2], axis=1)):
        raise ValueError("Duplicate evaluation user-song pair")
    user_ids, starts = np.unique(rows[:, 0], return_index=True)
    ends = np.append(starts[1:], len(rows))
    expected = p @ np.arange(1, 6)
    stars = rows[:, 2]
    error = expected - (stars + 1)
    nll = -np.log(np.maximum(p[np.arange(len(rows)), stars], 1e-30))
    users = []
    for user, start, end in zip(user_ids, starts, ends):
        star_is_relevant = stars[start:end] >= 3
        top = np.argsort(-expected[start:end], kind="stable")[:5]
        discount = 1.0 / np.log2(np.arange(2, len(top) + 2))
        ideal = discount[:star_is_relevant.sum()].sum()
        # Hash sorted candidate IDs and true labels so independently loaded
        # arm files cannot silently pair different evaluation targets.
        targets = np.ascontiguousarray(rows[start:end, 1:], dtype="<i8")
        record = {
            "user_index": int(user),
            "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
            "n_ratings": int(end - start),
            "mae": float(np.abs(error[start:end]).mean()),
            "mse": float(np.square(error[start:end]).mean()),
            "nll": float(nll[start:end].mean()),
            "ndcg5": float(star_is_relevant[top] @ discount / ideal) if ideal else None,
        }
        if original_user_ids is not None:
            record["original_user_id"] = int(original_user_ids[user])
        users.append(record)
    ranked = [record["ndcg5"] for record in users if record["ndcg5"] is not None]
    metrics = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "nll": float(nll.mean()),
        "ndcg5": float(np.mean(ranked)) if ranked else None,
        "ranking_users": len(ranked),
        "n_users": len(users),
        "n_ratings": len(rows),
    }
    return metrics, users


def paired_comparisons(by_arm, seed, k, reps=2000):
    """User-level intervals within each fitted seed, with paired users and songs."""
    if reps < 1 or set(by_arm) != set(ARMS):
        raise ValueError("Paired comparisons require three arms and positive repetitions")
    records = {arm: sorted(by_arm[arm], key=lambda u: u["user_index"]) for arm in ARMS}
    identifiers = {
        arm: [(u["user_index"], u.get("original_user_id"), u["target_sha256"])
              for u in records[arm]]
        for arm in ARMS
    }
    reference = identifiers["global"]
    if (not reference or len({r[0] for r in reference}) != len(reference)
            or any(ids != reference for ids in identifiers.values())):
        raise ValueError("Arms have different evaluation users, songs or labels")
    comparisons = []
    for left, right in CONTRASTS:
        delta = bootstrap_difference(records[right], records[left], seed, reps=reps)
        comparisons.append({
            "seed": int(seed), "k": int(k), "contrast": f"{left}_minus_{right}",
            "metrics": {
                metric: {"difference": result["C_minus_B"], "ci95": result["ci95"],
                         "users": result["users"]}
                for metric, result in delta.items()
            },
        })
    return comparisons


def _across_seeds(values):
    usable = [v for v in values if v is not None]
    return {"mean": float(np.mean(usable)) if usable else None,
            "sd": float(np.std(usable, ddof=1)) if len(usable) > 1 else None,
            "n_seeds": len(usable)}


def summarize_runs(results):
    """Descriptive summaries of arm metrics and within-seed differences by k."""
    by_k = {}
    for row in results:
        arms = by_k.setdefault(row["k"], {}).setdefault(row["seed"], {})
        if row["arm"] in arms:
            raise ValueError("Duplicate (k, seed, arm)")
        arms[row["arm"]] = row
    output = []
    for k, runs in sorted(by_k.items()):
        if any(set(arms) != set(ARMS) for arms in runs.values()):
            raise ValueError("Missing arm in paired seed summary")
        summary = {
            "k": int(k), "seeds": sorted(runs),
            "arms": {arm: {metric: _across_seeds([arms[arm][metric] for arms in runs.values()])
                           for metric in METRICS} for arm in ARMS},
            "contrasts": {},
        }
        for left, right in CONTRASTS:
            differences = {}
            for metric in METRICS:
                values = [(arms[left][metric], arms[right][metric]) for arms in runs.values()]
                differences[metric] = _across_seeds([a - b for a, b in values
                                                     if a is not None and b is not None])
            summary["contrasts"][f"{left}_minus_{right}"] = differences
        output.append(summary)
    return output
