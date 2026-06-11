"""
Exploratory Data Analysis engine for recommendation system.
Calculates distributions, sparsity, and long-tail analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

class NetflixEDA:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Set visualization style
        sns.set_theme(style="whitegrid")

    def analyze_rating_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate and plot rating distributions."""
        mean_rating = df['rating'].mean()
        var_rating = df['rating'].var()
        mode_rating = df['rating'].mode()[0]
        
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='rating', palette='viridis')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig(self.data_dir / 'rating_distribution.png')
        plt.close()
        
        return {'mean': mean_rating, 'variance': var_rating, 'mode': mode_rating}

    def calculate_sparsity(self, df: pd.DataFrame) -> float:
        """Calculate the global matrix sparsity."""
        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        n_ratings = len(df)
        
        if n_users == 0 or n_items == 0:
            return 100.0
            
        sparsity = (1 - (n_ratings / (n_users * n_items))) * 100
        return sparsity

    def long_tail_analysis(self, df: pd.DataFrame) -> float:
        """Analyze Pareto Principle (long tail) for movie ratings."""
        item_counts = df['item_id'].value_counts().sort_values(ascending=False)
        total_views = item_counts.sum()
        
        cumulative_views = item_counts.cumsum() / total_views
        # Find percentage of items making up 80% of views
        items_80_percent = len(cumulative_views[cumulative_views <= 0.8])
        pareto_percentage = (items_80_percent / len(item_counts)) * 100
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(cumulative_views)), cumulative_views.values)
        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.axvline(x=items_80_percent, color='r', linestyle='--')
        plt.title('Long Tail Analysis - Cumulative Rating Volume')
        plt.xlabel('Movie Rank')
        plt.ylabel('Cumulative % of Ratings')
        plt.savefig(self.data_dir / 'long_tail_analysis.png')
        plt.close()
        
        return pareto_percentage

    def run_full_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all EDA tasks and return metrics."""
        stats = self.analyze_rating_distribution(df)
        sparsity = self.calculate_sparsity(df)
        pareto_pct = self.long_tail_analysis(df)
        
        return {
            'rating_stats': stats,
            'sparsity_percentage': sparsity,
            'pareto_item_percentage': pareto_pct
        }
