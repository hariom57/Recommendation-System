"""
Explainability engine to provide human-readable justifications for recommendations.
"""

from typing import Dict, Any

class ExplanationEngine:
    def __init__(self, item_metadata: Dict[int, str] = None):
        self.item_metadata = item_metadata or {}

    def get_movie_name(self, item_id: int) -> str:
        return self.item_metadata.get(item_id, f"Movie {item_id}")

    def generate_cf_explanation(self, user_id: int, rec_item_id: int, sim_score: float, trigger_item_id: int) -> str:
        """Generate explanation for Collaborative Filtering based recommendations."""
        rec_name = self.get_movie_name(rec_item_id)
        trigger_name = self.get_movie_name(trigger_item_id)
        pct = min(99, int(sim_score * 100))
        
        return (f"Recommended '{rec_name}' because you rated '{trigger_name}' highly, "
                f"which has a {pct}% structural similarity score.")

    def generate_mf_explanation(self, user_id: int, rec_item_id: int, predicted_rating: float) -> str:
        """Generate explanation for Matrix Factorization based recommendations."""
        rec_name = self.get_movie_name(rec_item_id)
        return (f"Recommended '{rec_name}' because our latent feature model matches "
                f"it strongly with your taste profile (predicted rating: {predicted_rating:.1f}/5.0).")
