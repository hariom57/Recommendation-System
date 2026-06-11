"""
ExplanationEngine: generates human-readable recommendation explanations.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MOVIE_CACHE: Optional[dict] = None


def _load_movie_map() -> dict:
    global _MOVIE_CACHE
    if _MOVIE_CACHE is not None:
        return _MOVIE_CACHE
    titles_path = Path("data/netflix-prize-data/movie_titles.csv")
    if titles_path.exists():
        df = pd.read_csv(
            titles_path,
            encoding="latin-1",
            header=None,
            names=["item_id", "year", "title"],
        )
        _MOVIE_CACHE = dict(zip(df["item_id"].astype(int), df["title"].str.strip()))
    else:
        _MOVIE_CACHE = {}
    return _MOVIE_CACHE


class ExplanationEngine:
    """Generates textual explanations for recommendations."""

    def get_movie_name(self, item_id: int) -> str:
        movie_map = _load_movie_map()
        return movie_map.get(item_id, f"Movie #{item_id}")

    def generate_mf_explanation(self, user_id: int, item_id: int, score: float) -> str:
        title = self.get_movie_name(item_id)
        confidence = "strongly" if score >= 4.5 else "moderately" if score >= 3.5 else "slightly"
        return (
            f"Based on your viewing history, our model {confidence} predicts you will "
            f"enjoy '{title}' (predicted rating: {score:.2f}/5.0). "
            "This is derived from latent taste patterns shared with similar users."
        )

    def generate_cf_explanation(self, user_id: int, item_id: int, similar_items: list) -> str:
        title = self.get_movie_name(item_id)
        neighbours = ", ".join(f"'{self.get_movie_name(i)}'" for i in similar_items[:3])
        return (
            f"'{title}' is recommended because users who enjoyed {neighbours} "
            "also gave it high ratings — it shares strong structural similarities with your favourites."
        )
