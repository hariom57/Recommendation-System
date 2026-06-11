#!/usr/bin/env python3
"""
evaluate_models.py — Evaluate trained models on the held-out test split.

Usage
-----
  python evaluate_models.py [--k 10] [--test-sample 100000]

Outputs
-------
  reports/evaluation_results.csv   — RMSE and MAP@K for each model
  Console table with comparison
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    # Load test set
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        raise FileNotFoundError(
            "Test split not found. Please run train_pipeline.py first."
        )
    test = pd.read_parquet(test_path)
    if args.test_sample and len(test) > args.test_sample:
        test = test.sample(args.test_sample, random_state=42)
    logger.info("Test set: %d rows", len(test))

    results = []

    # Evaluate SVD
    svd_path = Path("models/svd.joblib")
    if svd_path.exists():
        logger.info("Loading SVD …")
        svd = joblib.load(svd_path)
        res = Evaluator.evaluate(svd, test, model_name="MatrixFactorizationSVD", k=args.k)
        results.append(res)
    else:
        logger.warning("SVD model not found at %s — skipping.", svd_path)

    # Evaluate CF (test sample to manage O(I^2) cost)
    cf_path = Path("models/item_cf.joblib")
    if cf_path.exists():
        logger.info("Loading ItemBasedCF …")
        cf = joblib.load(cf_path)
        cf_test = test.sample(min(30_000, len(test)), random_state=42)
        res = Evaluator.evaluate(cf, cf_test, model_name="ItemBasedCF", k=args.k)
        results.append(res)
    else:
        logger.warning("ItemBasedCF model not found at %s — skipping.", cf_path)

    if not results:
        logger.error("No models found. Train models first with train_pipeline.py")
        return

    # Print comparison table
    df = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 50)
    print("       MODEL COMPARISON")
    print("=" * 50)
    print(df.to_string())
    print("=" * 50 + "\n")

    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/evaluation_results.csv")
    logger.info("Saved to reports/evaluation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Cut-off rank for MAP@K (default: 10)")
    parser.add_argument("--test-sample", type=int, default=100_000,
                        help="Max test rows to use (default: 100K for speed)")
    main(parser.parse_args())
