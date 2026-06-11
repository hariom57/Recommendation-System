#!/usr/bin/env python3
"""
train_pipeline.py — End-to-end training pipeline for the Netflix Recommendation System.

Usage
-----
  python train_pipeline.py [--max-rows 5000000] [--cf-only] [--svd-only]

Steps
-----
1. Load & parse the Netflix Prize dataset
2. Downsample to a dense, tractable sub-matrix
3. Temporal train/test split (no data leakage)
4. Train ItemBasedCF → save to models/item_cf.joblib
5. Train MatrixFactorizationSVD → save to models/svd.joblib
6. Evaluate both on the held-out test set (RMSE & MAP@10)
7. Print a side-by-side comparison table
"""

import argparse
import time
import logging
from pathlib import Path

import pandas as pd

from src.data_processing import DataProcessor
from src.models import ItemBasedCF, MatrixFactorizationSVD
from src.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    dp = DataProcessor()

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading Netflix Prize dataset")
    logger.info("=" * 60)
    df = dp.load_data(max_rows=args.max_rows, use_cache=True)

    # ------------------------------------------------------------------ #
    # 2. Downsample                                                        #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STEP 2 — Downsampling")
    logger.info("=" * 60)
    df_dense = dp.downsample_data(df, min_user_ratings=50, min_item_ratings=50, max_users=50_000)

    # ------------------------------------------------------------------ #
    # 3. Train / test split                                                #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STEP 3 — Temporal train/test split")
    logger.info("=" * 60)
    train, test = dp.temporal_split(df_dense, test_ratio=0.1)
    dp.save_splits(train, test)

    results = []

    # ------------------------------------------------------------------ #
    # 4. ItemBasedCF                                                       #
    # ------------------------------------------------------------------ #
    if not args.svd_only:
        logger.info("=" * 60)
        logger.info("STEP 4 — Training ItemBasedCF")
        logger.info("=" * 60)
        t0 = time.time()
        cf_model = ItemBasedCF(k_neighbours=40)
        cf_model.fit(train)
        cf_model.save("models/item_cf.joblib")
        logger.info("ItemBasedCF trained in %.1f s", time.time() - t0)

        logger.info("Evaluating ItemBasedCF on test set …")
        # Use a sample of the test set for speed (full evaluation is slow with O(I^2) CF)
        test_sample = test.sample(min(50_000, len(test)), random_state=42)
        cf_results = Evaluator.evaluate(cf_model, test_sample, model_name="ItemBasedCF", k=10)
        results.append(cf_results)

    # ------------------------------------------------------------------ #
    # 5. MatrixFactorizationSVD                                            #
    # ------------------------------------------------------------------ #
    if not args.cf_only:
        logger.info("=" * 60)
        logger.info("STEP 5 — Training MatrixFactorizationSVD")
        logger.info("=" * 60)
        t0 = time.time()
        svd_model = MatrixFactorizationSVD(
            n_factors=50,
            n_epochs=20,
            lr=0.005,
            reg=0.02,
        )
        svd_model.fit(train)
        svd_model.save("models/svd.joblib")
        logger.info("SVD trained in %.1f s", time.time() - t0)

        logger.info("Evaluating MatrixFactorizationSVD on test set …")
        svd_results = Evaluator.evaluate(svd_model, test, model_name="MatrixFactorizationSVD", k=10)
        results.append(svd_results)

    # ------------------------------------------------------------------ #
    # 6. Results table                                                     #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    results_df = pd.DataFrame(results).set_index("model")
    print("\n" + results_df.to_string() + "\n")
    results_df.to_csv("reports/evaluation_results.csv")
    logger.info("Results saved to reports/evaluation_results.csv")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Netflix Recommendation System — Training Pipeline")
    parser.add_argument("--max-rows", type=int, default=5_000_000,
                        help="Max rows to load from the raw dataset (default: 5M). Use None for all 100M.")
    parser.add_argument("--cf-only", action="store_true", help="Train only ItemBasedCF")
    parser.add_argument("--svd-only", action="store_true", help="Train only MatrixFactorizationSVD")
    args = parser.parse_args()
    main(args)
