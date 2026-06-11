"""
Data processing module for Netflix recommendation system.
Handles simulated data generation, downsampling, and temporal splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(self.random_state)

    def generate_simulated_data(self, n_users: int = 10000, n_items: int = 2000, n_ratings: int = 100000) -> pd.DataFrame:
        """Simulate Netflix-like rating data."""
        users = np.random.randint(1, n_users + 1, n_ratings)
        items = np.random.randint(1, n_items + 1, n_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3], size=n_ratings)
        
        start_date = datetime(2020, 1, 1)
        timestamps = [start_date + timedelta(days=np.random.randint(0, 1000)) for _ in range(n_ratings)]
        
        df = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id'])
        return df

    def downsample_data(self, df: pd.DataFrame, top_users: int = 5000, top_items: int = 1000) -> pd.DataFrame:
        """Sample a dense subset of top users and items."""
        # Top items by rating count
        item_counts = df['item_id'].value_counts()
        top_item_ids = item_counts.head(top_items).index
        df_items = df[df['item_id'].isin(top_item_ids)]
        
        # Top users by rating count in the item-filtered dataset
        user_counts = df_items['user_id'].value_counts()
        top_user_ids = user_counts.head(top_users).index
        
        df_dense = df_items[df_items['user_id'].isin(top_user_ids)]
        return df_dense

    def temporal_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform a strict chronological train-test split."""
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        # Handle cold-start items in test
        train_items = set(train_df['item_id'])
        test_df = test_df[test_df['item_id'].isin(train_items)]
        
        return train_df, test_df
