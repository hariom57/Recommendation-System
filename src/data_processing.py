"""
DataProcessor: Loads and preprocesses the Netflix Prize dataset.

Dataset structure expected in data/netflix-prize-data/:
  - combined_data_1.txt
  - combined_data_2.txt
  - combined_data_3.txt
  - combined_data_4.txt
  - movie_titles.csv

Download from Kaggle: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles loading, parsing, downsampling, and temporal splitting of the Netflix Prize dataset."""

    DATA_DIR = Path("data/netflix-prize-data")
    PROCESSED_DIR = Path("data/processed")
    COMBINED_FILES = [
        "combined_data_1.txt",
        "combined_data_2.txt",
        "combined_data_3.txt",
        "combined_data_4.txt",
    ]
    MOVIE_TITLES_FILE = "movie_titles.csv"

    def __init__(self):
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(
        self,
        max_rows: Optional[int] = 5_000_000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load the Netflix Prize dataset from the raw text files.

        Parameters
        ----------
        max_rows : int or None
            Cap the number of rows parsed (None = all ~100 M rows).
            5 M is a sensible default for training on a laptop.
        use_cache : bool
            If True, read a previously cached parquet file when available.

        Returns
        -------
        pd.DataFrame with columns: user_id, item_id, rating, date
        """
        cache_path = self.PROCESSED_DIR / "ratings_raw.parquet"
        if use_cache and cache_path.exists():
            logger.info("Loading cached ratings from %s", cache_path)
            return pd.read_parquet(cache_path)

        self._check_data_files()

        frames = []
        rows_read = 0
        stop = False

        for fname in self.COMBINED_FILES:
            if stop:
                break
            fpath = self.DATA_DIR / fname
            logger.info("Parsing %s …", fpath)
            current_movie_id = None
            records = []

            with open(fpath, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.endswith(":"):
                        current_movie_id = int(line[:-1])
                        continue
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue
                    user_id, rating, date = parts
                    records.append(
                        (current_movie_id, int(user_id), int(rating), date.strip())
                    )
                    rows_read += 1
                    if max_rows and rows_read >= max_rows:
                        stop = True
                        break

            frames.append(
                pd.DataFrame(records, columns=["item_id", "user_id", "rating", "date"])
            )

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.astype({"user_id": "int32", "item_id": "int16", "rating": "int8"})

        logger.info("Loaded %d ratings for %d users and %d movies",
                    len(df), df["user_id"].nunique(), df["item_id"].nunique())

        df.to_parquet(cache_path, index=False)
        return df

    def load_movie_titles(self) -> pd.DataFrame:
        """
        Load movie_titles.csv.

        Returns
        -------
        pd.DataFrame with columns: item_id, year, title
        """
        fpath = self.DATA_DIR / self.MOVIE_TITLES_FILE
        if not fpath.exists():
            raise FileNotFoundError(
                f"movie_titles.csv not found at {fpath}. "
                "Please download the Netflix Prize data from Kaggle."
            )
        movies = pd.read_csv(
            fpath,
            encoding="latin-1",
            header=None,
            names=["item_id", "year", "title"],
        )
        movies["item_id"] = movies["item_id"].astype("int16")
        return movies

    def downsample_data(
        self,
        df: pd.DataFrame,
        min_user_ratings: int = 50,
        min_item_ratings: int = 50,
        max_users: Optional[int] = 50_000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Apply iterative density filter + optional random user cap to create
        a dense, tractable sub-matrix for training.
        """
        logger.info("Downsampling: starting with %d rows", len(df))

        # Iterative co-filtering
        for _ in range(5):
            n_before = len(df)
            user_counts = df["user_id"].value_counts()
            df = df[df["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)]
            item_counts = df["item_id"].value_counts()
            df = df[df["item_id"].isin(item_counts[item_counts >= min_item_ratings].index)]
            if len(df) == n_before:
                break

        # Cap users
        if max_users and df["user_id"].nunique() > max_users:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(df["user_id"].unique(), size=max_users, replace=False)
            df = df[df["user_id"].isin(chosen)]

        logger.info(
            "After downsampling: %d rows | %d users | %d movies",
            len(df), df["user_id"].nunique(), df["item_id"].nunique(),
        )
        return df.reset_index(drop=True)

    def temporal_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Per-user temporal split: the most recent `test_ratio` fraction of each
        user's ratings become the test set, the rest become train.
        This prevents data leakage from the future.
        """
        df = df.sort_values(["user_id", "date"])
        df["rank"] = df.groupby("user_id").cumcount(ascending=False)
        n_test_per_user = (df.groupby("user_id")["rating"].transform("count") * test_ratio).astype(int)
        test_mask = df["rank"] < n_test_per_user
        train = df[~test_mask].drop(columns=["rank"]).reset_index(drop=True)
        test = df[test_mask].drop(columns=["rank"]).reset_index(drop=True)
        logger.info("Train: %d rows | Test: %d rows", len(train), len(test))
        return train, test

    def save_splits(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        train.to_parquet(self.PROCESSED_DIR / "train.parquet", index=False)
        test.to_parquet(self.PROCESSED_DIR / "test.parquet", index=False)
        logger.info("Saved train/test splits to %s", self.PROCESSED_DIR)

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(self.PROCESSED_DIR / "train.parquet")
        test = pd.read_parquet(self.PROCESSED_DIR / "test.parquet")
        return train, test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_data_files(self):
        if not self.DATA_DIR.exists():
            raise FileNotFoundError(
                f"\nData directory '{self.DATA_DIR}' not found.\n"
                "Please download the Netflix Prize dataset from Kaggle:\n"
                "  https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data\n"
                "and unzip it into:  data/netflix-prize-data/"
            )
        missing = [f for f in self.COMBINED_FILES if not (self.DATA_DIR / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing Netflix data files: {missing}\n"
                "Check that the Kaggle download is complete and placed in data/netflix-prize-data/"
            )
