import numpy as np


class EASE:
    def __init__(self, reg=100.0):
        self.reg = reg

    def fit(self, X):
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.solve(G, np.eye(G.shape[0]))


        self.B = -P / np.diag(P)[None, :]
        np.fill_diagonal(self.B, 0.0)

        return self

    def scores_for_user(self, user_idx, X):
        return np.asarray(
            X.getrow(user_idx) @ self.B
        ).ravel()
