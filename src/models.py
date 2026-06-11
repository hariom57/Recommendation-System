"""
Object-Oriented Modeling Core containing Matrix Factorization and Item-Based CF.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, train_set: pd.DataFrame) -> None:
        pass
        
    @abstractmethod
    def predict(self, user: int, item: int) -> float:
        pass
        
    @abstractmethod
    def recommend(self, user: int, k: int = 10) -> List[Tuple[int, float]]:
        pass

class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering using adjusted cosine similarity."""
    def __init__(self):
        self.item_sim_matrix: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_means: Optional[pd.Series] = None
        
    def fit(self, train_set: pd.DataFrame) -> None:
        self.user_item_matrix = train_set.pivot(index='user_id', columns='item_id', values='rating')
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Mean-centered ratings for adjusted cosine similarity
        centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0)
        
        # Calculate item-item similarity matrix
        self.item_sim_matrix = centered_matrix.corr(method='pearson', min_periods=2)
        self.item_sim_matrix.fillna(0, inplace=True)
        
    def predict(self, user: int, item: int) -> float:
        if self.user_item_matrix is None or self.item_sim_matrix is None or self.user_means is None:
            raise ValueError("Model must be fitted before prediction.")
            
        if user not in self.user_item_matrix.index or item not in self.item_sim_matrix.columns:
            return 3.0 # Fallback for complete cold start
            
        user_ratings = self.user_item_matrix.loc[user].dropna()
        if user_ratings.empty:
            return self.user_means.get(user, 3.0)
            
        similarities = self.item_sim_matrix.loc[item, user_ratings.index]
        
        # Only use positive similarities for prediction
        pos_sim_mask = similarities > 0
        if not pos_sim_mask.any():
            return self.user_means.get(user, 3.0)
            
        weighted_sum = (similarities[pos_sim_mask] * user_ratings[pos_sim_mask]).sum()
        sum_of_weights = similarities[pos_sim_mask].sum()
        
        if sum_of_weights == 0:
            return self.user_means.get(user, 3.0)
            
        return weighted_sum / sum_of_weights

    def recommend(self, user: int, k: int = 10) -> List[Tuple[int, float]]:
        if self.user_item_matrix is None or self.item_sim_matrix is None:
            raise ValueError("Model must be fitted before recommendation.")
            
        if user not in self.user_item_matrix.index:
            return []
            
        user_ratings = self.user_item_matrix.loc[user].dropna()
        unrated_items = self.item_sim_matrix.columns.difference(user_ratings.index)
        
        predictions = [(item, self.predict(user, item)) for item in unrated_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]


class MatrixFactorizationSVD(BaseRecommender):
    """Regularized Matrix Factorization using SGD (Funk SVD)."""
    def __init__(self, n_factors: int = 20, learning_rate: float = 0.005, reg: float = 0.02, n_epochs: int = 20):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.global_mean = 0.0
        self.P: Dict[int, np.ndarray] = {}
        self.Q: Dict[int, np.ndarray] = {}
        self.users_list: List[int] = []
        self.items_list: List[int] = []
        
    def fit(self, train_set: pd.DataFrame) -> None:
        self.global_mean = train_set['rating'].mean()
        self.users_list = train_set['user_id'].unique().tolist()
        self.items_list = train_set['item_id'].unique().tolist()
        
        # Initialize latent vectors
        for u in self.users_list:
            self.P[u] = np.random.normal(scale=1./self.n_factors, size=self.n_factors)
        for i in self.items_list:
            self.Q[i] = np.random.normal(scale=1./self.n_factors, size=self.n_factors)
            
        train_data = train_set[['user_id', 'item_id', 'rating']].values
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)
            for u, i, r in train_data:
                # Calculate error
                pred = self.P[u].dot(self.Q[i])
                e = r - pred
                
                # Update latent vectors using SGD
                P_u_update = self.P[u] + self.lr * (e * self.Q[i] - self.reg * self.P[u])
                Q_i_update = self.Q[i] + self.lr * (e * self.P[u] - self.reg * self.Q[i])
                
                self.P[u] = P_u_update
                self.Q[i] = Q_i_update
                
    def predict(self, user: int, item: int) -> float:
        if user in self.P and item in self.Q:
            return float(self.P[user].dot(self.Q[item]))
        return self.global_mean

    def recommend(self, user: int, k: int = 10) -> List[Tuple[int, float]]:
        if user not in self.P:
            # Cold start fallback
            return []
            
        predictions = []
        user_vec = self.P[user]
        for item in self.items_list:
            pred = float(user_vec.dot(self.Q[item]))
            predictions.append((item, pred))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]
