"""
FastAPI production application for the Netflix Recommendation System.

Features
--------
- Loads pre-trained models from disk via joblib at startup
- Serves personalized recommendations via MatrixFactorizationSVD
- Cold-start fallback: returns globally popular movies for unknown users
- Deprecation-safe: uses the modern `lifespan` context manager (FastAPI ≥ 0.93)
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.explain import ExplanationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

SVD_PATH = "models/svd.joblib"
CF_PATH = "models/item_cf.joblib"
TRAIN_PATH = "data/processed/train.parquet"

_svd_model = None
_cf_model = None
_popular_movies: list = []
_explain_engine = ExplanationEngine()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and pre-compute popular-movie fallback list at startup."""
    global _svd_model, _cf_model, _popular_movies

    # Load SVD model
    if Path(SVD_PATH).exists():
        logger.info("Loading SVD model from %s …", SVD_PATH)
        _svd_model = joblib.load(SVD_PATH)
        logger.info("SVD model loaded.")
    else:
        logger.warning(
            "SVD model not found at %s. Run train_pipeline.py first. "
            "API will serve cold-start recommendations only.", SVD_PATH
        )

    # Load CF model (optional — used by /recommend/cf endpoint)
    if Path(CF_PATH).exists():
        logger.info("Loading ItemBasedCF model from %s …", CF_PATH)
        _cf_model = joblib.load(CF_PATH)
        logger.info("ItemBasedCF model loaded.")

    # Compute popular movies from training data for cold-start
    if Path(TRAIN_PATH).exists():
        logger.info("Computing popular-movie fallback from %s …", TRAIN_PATH)
        train = pd.read_parquet(TRAIN_PATH, columns=["item_id", "rating"])
        popular = (
            train.groupby("item_id")["rating"]
            .agg(["count", "mean"])
            .query("count >= 1000")
            .sort_values("mean", ascending=False)
            .head(50)
            .reset_index()
        )
        _popular_movies = popular["item_id"].tolist()
        logger.info("Popular-movie fallback list computed (%d movies).", len(_popular_movies))
    else:
        # Hardcoded fallback if training data is also absent
        _popular_movies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        logger.warning("Training data not found; using placeholder popular-movie list.")

    yield  # ← application runs here

    # Teardown (optional cleanup)
    logger.info("Shutting down recommendation API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Netflix Recommendation API",
    version="2.0.0",
    description=(
        "Production recommendation engine trained on the Netflix Prize dataset. "
        "Uses Funk SVD (RMSE-optimised) and Item-Based CF (MAP@10-optimised)."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RecRequest(BaseModel):
    user_id: int
    top_k: int = 10
    model: str = "svd"  # "svd" | "cf"


class RecResponseItem(BaseModel):
    item_id: int
    title: str
    predicted_rating: float
    explanation: str


class RecResponse(BaseModel):
    user_id: int
    model_used: str
    is_cold_start: bool
    recommendations: List[RecResponseItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "svd_loaded": _svd_model is not None,
        "cf_loaded": _cf_model is not None,
        "popular_movies_count": len(_popular_movies),
    }


@app.post("/api/v1/recommend", response_model=RecResponse)
async def get_recommendations(req: RecRequest):
    """
    Get Top-K recommendations for a user.

    - model="svd"  → MatrixFactorizationSVD (lower RMSE, fast inference)
    - model="cf"   → ItemBasedCF           (higher MAP@10, explainable)
    """
    top_k = max(1, min(req.top_k, 50))
    active_model = _svd_model if req.model == "svd" else _cf_model

    # ── Cold-start fallback ──────────────────────────────────────────────
    cold_start = active_model is None

    if not cold_start:
        try:
            recs = active_model.recommend(req.user_id, k=top_k)
        except Exception as exc:
            logger.error("recommend() failed: %s", exc)
            recs = []
        if not recs:
            cold_start = True

    if cold_start:
        items = []
        for iid in _popular_movies[:top_k]:
            title = _explain_engine.get_movie_name(iid)
            items.append(RecResponseItem(
                item_id=iid,
                title=title,
                predicted_rating=0.0,
                explanation=(
                    "Trending globally: recommended because this movie is consistently "
                    "rated highly by a large number of Netflix users."
                ),
            ))
        return RecResponse(
            user_id=req.user_id,
            model_used="cold_start_popular",
            is_cold_start=True,
            recommendations=items,
        )

    # ── Personalised recommendations ────────────────────────────────────
    items = []
    for item_id, score in recs:
        title = _explain_engine.get_movie_name(item_id)
        if req.model == "svd":
            explanation = _explain_engine.generate_mf_explanation(req.user_id, item_id, score)
        else:
            explanation = _explain_engine.generate_cf_explanation(req.user_id, item_id, [])
        items.append(RecResponseItem(
            item_id=item_id,
            title=title,
            predicted_rating=round(score, 2),
            explanation=explanation,
        ))

    return RecResponse(
        user_id=req.user_id,
        model_used=req.model,
        is_cold_start=False,
        recommendations=items,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
