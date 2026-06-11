"""
Evaluator: computes RMSE and MAP@K on a held-out test set.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Evaluator:
    """Evaluation suite for recommendation models."""

    # ------------------------------------------------------------------
    # Rating-prediction metric
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(model, test_df: pd.DataFrame) -> float:
        """
        Root Mean Squared Error on explicit rating predictions.

        Parameters
        ----------
        model    : fitted model with a .predict(user_id, item_id) method
        test_df  : DataFrame with columns user_id, item_id, rating

        Returns
        -------
        float — RMSE
        """
        errors = []
        for _, row in test_df.iterrows():
            pred = model.predict(int(row["user_id"]), int(row["item_id"]))
            errors.append((pred - row["rating"]) ** 2)
        rmse_val = float(np.sqrt(np.mean(errors)))
        logger.info("RMSE = %.4f  (n=%d)", rmse_val, len(errors))
        return rmse_val

    # ------------------------------------------------------------------
    # Ranking metric
    # ------------------------------------------------------------------

    @staticmethod
    def map_at_k(model, test_df: pd.DataFrame, k: int = 10, threshold: float = 4.0) -> float:
        """
        Mean Average Precision @ K.

        A rating >= `threshold` is considered a relevant item.

        Parameters
        ----------
        model     : fitted model with a .recommend(user_id, k) → [(item_id, score)]
        test_df   : held-out ratings DataFrame
        k         : cut-off rank
        threshold : minimum rating counted as relevant

        Returns
        -------
        float — MAP@K
        """
        # Build ground-truth: per-user set of relevant items
        relevant = (
            test_df[test_df["rating"] >= threshold]
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        ap_scores = []
        for user_id, rel_items in relevant.items():
            recs = model.recommend(int(user_id), k=k)
            if not recs:
                ap_scores.append(0.0)
                continue
            rec_ids = [iid for iid, _ in recs]
            hits = 0
            precision_sum = 0.0
            for rank, iid in enumerate(rec_ids, start=1):
                if iid in rel_items:
                    hits += 1
                    precision_sum += hits / rank
            ap = precision_sum / min(len(rel_items), k) if rel_items else 0.0
            ap_scores.append(ap)

        map_val = float(np.mean(ap_scores)) if ap_scores else 0.0
        logger.info("MAP@%d = %.4f  (n_users=%d)", k, map_val, len(ap_scores))
        return map_val

    # ------------------------------------------------------------------
    # Full evaluation report
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(model, test_df: pd.DataFrame, model_name: str = "Model", k: int = 10) -> dict:
        logger.info("Evaluating %s …", model_name)
        rmse = Evaluator.rmse(model, test_df)
        map_k = Evaluator.map_at_k(model, test_df, k=k)
        results = {
            "model": model_name,
            "rmse": round(rmse, 4),
            f"map@{k}": round(map_k, 4),
        }
        logger.info("Results → %s", results)
        return results
