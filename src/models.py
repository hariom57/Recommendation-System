"""
Recommendation models:
  - ItemBasedCF  : Item-Based Collaborative Filtering (cosine similarity)
  - MatrixFactorizationSVD : Funk SVD via SGD

Both models support:
  - fit(train_df)
  - predict(user_id, item_id)  → float
  - recommend(user_id, k)      → [(item_id, score), ...]
  - save(path) / load(path)    via joblib
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering.

    Uses pre-computed item-item cosine similarity on the user-item rating matrix.
    Prediction for (user u, item i) is the rating-weighted average of the k nearest
    neighbours of item i that user u has already rated.

    Optimizes MAP@K via high-quality ranked retrieval.
    """

    def __init__(self, k_neighbours: int = 40):
        self.k_neighbours = k_neighbours
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None  # users × items
        self.item_means: Optional[pd.Series] = None
        self.item_index: Optional[dict] = None   # item_id → col index
        self.user_index: Optional[dict] = None   # user_id → row index
        self._is_fitted = False

    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "ItemBasedCF":
        logger.info("ItemBasedCF: building user-item matrix …")
        # Build pivot: rows=users, cols=items, values=mean-centered ratings
        matrix = train_df.pivot_table(
            index="user_id", columns="item_id", values="rating", aggfunc="mean"
        )
        self.item_means = matrix.mean(axis=0)
        centered = matrix.subtract(self.item_means, axis=1).fillna(0)

        self.user_item_matrix = matrix
        self.item_index = {iid: idx for idx, iid in enumerate(matrix.columns)}
        self.user_index = {uid: idx for idx, uid in enumerate(matrix.index)}

        logger.info("ItemBasedCF: computing item-item cosine similarity (%d items) …",
                    len(self.item_index))
        sparse = csr_matrix(centered.values.T)  # items × users
        self.similarity_matrix = cosine_similarity(sparse, dense_output=True)
        self._is_fitted = True
        logger.info("ItemBasedCF: fit complete.")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        self._check_fitted()
        if user_id not in self.user_index or item_id not in self.item_index:
            # Fall back to item mean or global mean
            if item_id in self.item_index:
                return float(self.item_means.iloc[self.item_index[item_id]])
            return float(self.item_means.mean())

        item_col = self.item_index[item_id]
        user_row = self.user_index[user_id]

        # Ratings this user has given (non-NaN)
        user_ratings = self.user_item_matrix.iloc[user_row]
        rated_mask = user_ratings.notna()
        rated_items = user_ratings[rated_mask]

        # Similarities between target item and rated items
        rated_cols = [self.item_index[i] for i in rated_items.index if i in self.item_index]
        if not rated_cols:
            return float(self.item_means.iloc[item_col])

        sims = self.similarity_matrix[item_col, rated_cols]
        ratings = rated_items.values[:len(rated_cols)]

        # Take top-k neighbours
        top_k = min(self.k_neighbours, len(sims))
        idx = np.argpartition(sims, -top_k)[-top_k:]
        sims_k = sims[idx]
        ratings_k = ratings[idx]

        denom = np.abs(sims_k).sum()
        if denom == 0:
            return float(self.item_means.iloc[item_col])
        return float(np.dot(sims_k, ratings_k) / denom)

    def recommend(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        self._check_fitted()
        if user_id not in self.user_index:
            return []

        user_ratings = self.user_item_matrix.iloc[self.user_index[user_id]]
        unrated_items = user_ratings[user_ratings.isna()].index.tolist()

        scores = [(iid, self.predict(user_id, iid)) for iid in unrated_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # ------------------------------------------------------------------

    def save(self, path: str = "models/item_cf.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("ItemBasedCF saved to %s", path)

    @classmethod
    def load(cls, path: str = "models/item_cf.joblib") -> "ItemBasedCF":
        model = joblib.load(path)
        logger.info("ItemBasedCF loaded from %s", path)
        return model

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("ItemBasedCF is not fitted. Call .fit() first.")


# ======================================================================


class MatrixFactorizationSVD:
    """
    Funk SVD (SGD-based matrix factorization).

    Minimizes RMSE via stochastic gradient descent on explicit ratings.
    Incorporates user bias, item bias, and global mean.

    Optimizes RMSE — the primary metric for rating-prediction tasks.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

        # Learned parameters (set during fit)
        self.global_mean: float = 0.0
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        # Mappings
        self.user_map: Optional[dict] = None   # user_id → row idx
        self.item_map: Optional[dict] = None   # item_id → col idx
        self.item_ids: Optional[np.ndarray] = None  # col idx → item_id

        self._is_fitted = False

    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "MatrixFactorizationSVD":
        rng = np.random.default_rng(self.seed)

        users = train_df["user_id"].unique()
        items = train_df["item_id"].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: i for i, it in enumerate(items)}
        self.item_ids = items

        n_users = len(users)
        n_items = len(items)

        self.global_mean = train_df["rating"].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors))

        u_idx = train_df["user_id"].map(self.user_map).values
        i_idx = train_df["item_id"].map(self.item_map).values
        ratings = train_df["rating"].values.astype(np.float32)

        logger.info("MatrixFactorizationSVD: training %d epochs, %d samples …",
                    self.n_epochs, len(ratings))

        order = np.arange(len(ratings))
        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            epoch_loss = 0.0

            for idx in order:
                u, i, r = u_idx[idx], i_idx[idx], ratings[idx]
                pred = (
                    self.global_mean
                    + self.user_biases[u]
                    + self.item_biases[i]
                    + self.user_factors[u] @ self.item_factors[i]
                )
                err = r - pred
                epoch_loss += err ** 2

                # Update biases
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])

                # Update latent factors
                uf_old = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * uf_old - self.reg * self.item_factors[i])

            rmse = np.sqrt(epoch_loss / len(ratings))
            logger.info("  Epoch %d/%d — Train RMSE: %.4f", epoch + 1, self.n_epochs, rmse)

        self._is_fitted = True
        logger.info("MatrixFactorizationSVD: training complete.")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        self._check_fitted()
        u = self.user_map.get(user_id)
        i = self.item_map.get(item_id)
        if u is None or i is None:
            return float(self.global_mean)
        pred = (
            self.global_mean
            + self.user_biases[u]
            + self.item_biases[i]
            + self.user_factors[u] @ self.item_factors[i]
        )
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        self._check_fitted()
        u = self.user_map.get(user_id)
        if u is None:
            return []
        # Vectorised: score all items at once
        scores = (
            self.global_mean
            + self.user_biases[u]
            + self.item_biases
            + self.item_factors @ self.user_factors[u]
        )
        scores = np.clip(scores, 1.0, 5.0)
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top_k_idx]

    # ------------------------------------------------------------------

    def save(self, path: str = "models/svd.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("MatrixFactorizationSVD saved to %s", path)

    @classmethod
    def load(cls, path: str = "models/svd.joblib") -> "MatrixFactorizationSVD":
        model = joblib.load(path)
        logger.info("MatrixFactorizationSVD loaded from %s", path)
        return model

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("MatrixFactorizationSVD is not fitted. Call .fit() first.")
