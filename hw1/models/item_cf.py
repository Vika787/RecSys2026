import numpy as np


def make_item_similarity(cosine, common, min_common=3, shrinkage=10):
    weights = np.divide(
        common,
        common + shrinkage,
        out=np.zeros_like(common, dtype=np.float32),
        where=(common + shrinkage) > 0
    )

    similarity = cosine * weights
    similarity[common < min_common] = 0.0
    np.fill_diagonal(similarity, 0.0)

    return similarity


def item_cf_scores(user_idx, similarity, R, user_means):
    row = R.getrow(user_idx)
    seen_items = row.indices
    seen_ratings = row.data

    similarities = np.maximum(
        similarity[:, seen_items], 0
    )

    deviations = seen_ratings - user_means[user_idx]

    numerator = similarities @ deviations
    denominator = similarities.sum(axis=1)

    scores = np.full(R.shape[1], np.nan)
    valid = denominator > 0

    scores[valid] = (
        user_means[user_idx]
        + numerator[valid] / denominator[valid]
    )

    scores[seen_items] = np.nan

    return scores


def recommend_item_cf(
    user_idx, similarity, k=10,
    R=None, user_means=None,
    popularity_order=None
):
    if R is None or user_means is None or popularity_order is None:
        raise ValueError("R, user_means and popularity_order are required")

    scores = item_cf_scores(user_idx, similarity, R, user_means)

    candidates = np.flatnonzero(np.isfinite(scores))
    ranked = candidates[
        np.lexsort((candidates, -scores[candidates]))
    ]

    top = ranked[:k].tolist()

    if len(top) < k:
        seen = set(R.getrow(user_idx).indices)
        selected = set(top)

        for item in popularity_order:
            if item not in seen and item not in selected:
                top.append(int(item))
                if len(top) == k:
                    break

    return np.array(top, dtype=int)
