import numpy as np
import torch
from torch import nn


class TimeSVDpp(nn.Module):
    def __init__(
        self, n_users, n_items, n_bins, history,
        user_mean_days, origin_day, bin_days,
        global_mean, n_factors=16
    ):
        super().__init__()

        self.n_items = n_items
        self.n_bins = n_bins
        self.origin_day = origin_day
        self.bin_days = bin_days
        self.user_mean_days = user_mean_days

        self.register_buffer("history", history)

        self.global_mean = nn.Parameter(
            torch.tensor(float(global_mean))
        )

        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)

        self.bu_drift = nn.Embedding(n_users, 1)
        self.bi_time = nn.Embedding(n_items * n_bins, 1)

        self.p = nn.Embedding(n_users, n_factors)
        self.q = nn.Embedding(n_items, n_factors)

        self.p_drift = nn.Embedding(n_users, n_factors)

        self.y = nn.Embedding(n_items, n_factors)

        for emb in [self.bu, self.bi, self.bu_drift, self.bi_time]:
            nn.init.zeros_(emb.weight)

        for emb in [self.p, self.q, self.y]:
            nn.init.normal_(emb.weight, std=0.05)

        nn.init.zeros_(self.p_drift.weight)

    def forward(self, users, items, dev, bins):
        history_factors = torch.sparse.mm(
            self.history, self.y.weight
        )

        user_factors = (
            self.p(users)
            + self.p_drift(users) * dev[:, None]
            + history_factors[users]
        )

        predictions = (
            self.global_mean
            + self.bu(users).squeeze(-1)
            + self.bu_drift(users).squeeze(-1) * dev
            + self.bi(items).squeeze(-1)
            + self.bi_time(items * self.n_bins + bins).squeeze(-1)
            + (user_factors * self.q(items)).sum(dim=1)
        )

        return predictions

    def prepare_inference(self):
        with torch.no_grad():
            self.history_cache = torch.sparse.mm(
                self.history, self.y.weight
            )

    def scores_for_user(self, user_idx, day):
        dev = day - self.user_mean_days[user_idx]
        dev = float(np.sign(dev) * abs(dev) ** 0.4)

        bin_idx = int(np.clip(
            (day - self.origin_day) // self.bin_days,
            0, self.n_bins - 1
        ))

        with torch.no_grad():
            user_factors = (
                self.p.weight[user_idx]
                + self.p_drift.weight[user_idx] * dev
                + self.history_cache[user_idx]
            )

            item_time_indices = (
                torch.arange(self.n_items) * self.n_bins
                + bin_idx
            )

            scores = (
                self.global_mean
                + self.bu.weight[user_idx, 0]
                + self.bu_drift.weight[user_idx, 0] * dev
                + self.bi.weight[:, 0]
                + self.bi_time(item_time_indices).squeeze(-1)
                + self.q.weight @ user_factors
            )

        return scores.numpy().copy()
