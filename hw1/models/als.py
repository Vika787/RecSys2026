import numpy as np


class ExplicitALS:
    def __init__(self, n_factors=16, n_iters=5, reg=1.0, seed=42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.seed = seed

    def fit(self, R):
        R = R.tocsr().astype(np.float64)
        R_csc = R.tocsc()

        n_users, n_items = R.shape
        rng = np.random.default_rng(self.seed)

        self.user_factors = rng.normal(
            0, 0.1, size=(n_users, self.n_factors)
        )
        self.item_factors = rng.normal(
            0, 0.1, size=(n_items, self.n_factors)
        )

        eye = np.eye(self.n_factors)
        observed = R.tocoo()
        self.train_rmse = []

        for iteration in range(self.n_iters):

            for u in range(n_users):
                start, end = R.indptr[u:u + 2]
                items = R.indices[start:end]
                ratings = R.data[start:end]

                Q = self.item_factors[items]

                self.user_factors[u] = np.linalg.solve(
                    Q.T @ Q + self.reg * eye,
                    Q.T @ ratings
                )

            for i in range(n_items):
                start, end = R_csc.indptr[i:i + 2]
                users = R_csc.indices[start:end]
                ratings = R_csc.data[start:end]

                P = self.user_factors[users]

                self.item_factors[i] = np.linalg.solve(
                    P.T @ P + self.reg * eye,
                    P.T @ ratings
                )

            predictions = self.predict_pairs(
                observed.row, observed.col
            )
            rmse = np.sqrt(np.mean(
                (observed.data - predictions) ** 2
            ))
            self.train_rmse.append(rmse)

        return self

    def predict_pairs(self, user_idx, item_idx):
        return np.sum(
            self.user_factors[user_idx] *
            self.item_factors[item_idx],
            axis=1
        )

    def scores_for_user(self, user_idx):
        return self.item_factors @ self.user_factors[user_idx]
