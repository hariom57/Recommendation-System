"""
FastAPI production application for the recommendation system endpoint.
Features cold-start fallback routing.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import numpy as np

# In a real scenario, models would be loaded from disk (e.g. pickle or joblib).
# For this script, we'll instantiate mock/dummy states for demonstration of the API structure.
from src.models import MatrixFactorizationSVD
from src.explain import ExplanationEngine

app = FastAPI(title="Netflix Recommender API", version="1.0.0")

# Mock database and models
model = MatrixFactorizationSVD(n_factors=10)
explain_engine = ExplanationEngine()
POPULAR_MOVIES = []

class RecRequest(BaseModel):
    user_id: int
    top_k: int = 10

class RecResponseItem(BaseModel):
    item_id: int
    title: str
    explanation: str

class RecResponse(BaseModel):
    user_id: int
    recommendations: List[RecResponseItem]

@app.on_event("startup")
async def load_models():
    """Load pre-trained models and compute global fallbacks."""
    global POPULAR_MOVIES
    # Simulate loading global popular movies for cold start
    POPULAR_MOVIES = [
        {"item_id": 1, "title": "The Shawshank Redemption"},
        {"item_id": 2, "title": "The Godfather"},
        {"item_id": 3, "title": "The Dark Knight"},
        {"item_id": 4, "title": "Pulp Fiction"},
        {"item_id": 5, "title": "Forrest Gump"}
    ]

@app.post("/api/v1/recommend", response_model=RecResponse)
async def get_recommendations(req: RecRequest):
    try:
        recs = model.recommend(req.user_id, k=req.top_k)
        
        response_items = []
        
        # Cold-Start Fallback Routing mechanism
        if not recs:
            # User is unknown or model returned no recs, fallback to popular
            for i, movie in enumerate(POPULAR_MOVIES[:req.top_k]):
                response_items.append(RecResponseItem(
                    item_id=movie["item_id"],
                    title=movie["title"],
                    explanation="Recommended because it is currently trending globally among all users."
                ))
            return RecResponse(user_id=req.user_id, recommendations=response_items)

        # Generate personalized recommendations
        for item_id, score in recs:
            title = explain_engine.get_movie_name(item_id)
            exp = explain_engine.generate_mf_explanation(req.user_id, item_id, score)
            response_items.append(RecResponseItem(
                item_id=item_id,
                title=title,
                explanation=exp
            ))
            
        return RecResponse(user_id=req.user_id, recommendations=response_items)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
