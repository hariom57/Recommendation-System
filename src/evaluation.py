"""
Evaluation metrics suite for the recommendation system.
Implements RMSE and MAP@10 from scratch.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from .models import BaseRecommender

class Evaluator:
    @staticmethod
    def calculate_rmse(model: BaseRecommender, test_set: pd.DataFrame) -> float:
        """Calculate Root Mean Squared Error."""
        squared_errors = []
        for _, row in test_set.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id'])
            r_true = float(row['rating'])
            r_pred = model.predict(u, i)
            squared_errors.append((r_true - r_pred) ** 2)
            
        if not squared_errors:
            return 0.0
            
        return np.sqrt(np.mean(squared_errors))

    @staticmethod
    def calculate_map_at_k(model: BaseRecommender, test_set: pd.DataFrame, k: int = 10, threshold: float = 3.5) -> float:
        """Calculate Mean Average Precision at K."""
        # Group true relevant items by user
        user_true_items = {}
        for _, row in test_set.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id'])
            r = float(row['rating'])
            if r >= threshold:
                if u not in user_true_items:
                    user_true_items[u] = set()
                user_true_items[u].add(i)
                
        average_precisions = []
        
        for u, true_items in user_true_items.items():
            if not true_items:
                continue
                
            # Get recommendations for the user
            recs = model.recommend(u, k=k)
            if not recs:
                average_precisions.append(0.0)
                continue
                
            rec_items = [item_id for item_id, _ in recs]
            
            hits = 0
            sum_precs = 0.0
            
            for i, item_id in enumerate(rec_items):
                if item_id in true_items:
                    hits += 1
                    sum_precs += hits / (i + 1.0)
                    
            if hits > 0:
                ap = sum_precs / min(len(true_items), k)
            else:
                ap = 0.0
                
            average_precisions.append(ap)
            
        if not average_precisions:
            return 0.0
            
        return np.mean(average_precisions)
