"""
NetflixEDA: Exploratory Data Analysis for the Netflix Prize dataset.

Generates and saves publication-quality plots to reports/figures/.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend — safe in scripts & notebooks
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")


class NetflixEDA:
    """Full EDA suite for the Netflix Prize interaction data."""

    def __init__(self, figures_dir: Path = FIGURES_DIR):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------

    def run_full_analysis(self, df: pd.DataFrame, movies_df: pd.DataFrame = None) -> dict:
        """
        Run all analyses and return a dict of key statistics.

        Parameters
        ----------
        df       : ratings DataFrame (user_id, item_id, rating, date)
        movies_df: optional movie metadata DataFrame (item_id, year, title)
        """
        stats = {}
        stats.update(self.basic_statistics(df))
        self.plot_rating_distribution(df)
        self.plot_ratings_over_time(df)
        self.plot_user_activity(df)
        self.plot_item_popularity(df)
        self.plot_sparsity_heatmap(df)
        self.plot_rating_by_year(df, movies_df)
        self.plot_user_item_degree_distribution(df)
        return stats

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def basic_statistics(self, df: pd.DataFrame) -> dict:
        n_users = df["user_id"].nunique()
        n_items = df["item_id"].nunique()
        n_ratings = len(df)
        density = n_ratings / (n_users * n_items)
        sparsity = 1 - density
        avg_rating = df["rating"].mean()
        std_rating = df["rating"].std()
        avg_per_user = df.groupby("user_id")["rating"].count().mean()
        avg_per_item = df.groupby("item_id")["rating"].count().mean()

        stats = {
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": n_ratings,
            "matrix_density": round(density, 6),
            "matrix_sparsity": round(sparsity, 6),
            "avg_rating": round(avg_rating, 3),
            "std_rating": round(std_rating, 3),
            "avg_ratings_per_user": round(avg_per_user, 1),
            "avg_ratings_per_item": round(avg_per_item, 1),
        }
        logger.info(
            "Dataset: %d users | %d items | %d ratings | sparsity=%.4f | avg_rating=%.2f",
            n_users, n_items, n_ratings, sparsity, avg_rating,
        )
        return stats

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_rating_distribution(self, df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Raw counts
        counts = df["rating"].value_counts().sort_index()
        axes[0].bar(counts.index, counts.values, color=PALETTE[0], edgecolor="white", width=0.7)
        axes[0].set_xlabel("Star Rating")
        axes[0].set_ylabel("Number of Ratings")
        axes[0].set_title("Rating Frequency Distribution")
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

        # Percentage
        pct = counts / counts.sum() * 100
        axes[1].bar(pct.index, pct.values, color=PALETTE[1], edgecolor="white", width=0.7)
        axes[1].set_xlabel("Star Rating")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("Rating Distribution (Percentage)")
        for i, v in zip(pct.index, pct.values):
            axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

        fig.suptitle("Netflix Prize — Rating Distributions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "rating_distribution.png")

    def plot_ratings_over_time(self, df: pd.DataFrame):
        if "date" not in df.columns:
            return
        monthly = df.set_index("date").resample("ME")["rating"].count().reset_index()
        monthly.columns = ["month", "count"]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(monthly["month"], monthly["count"], alpha=0.3, color=PALETTE[2])
        ax.plot(monthly["month"], monthly["count"], color=PALETTE[2], linewidth=2)
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Ratings")
        ax.set_title("Monthly Rating Volume Over Time")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
        plt.tight_layout()
        self._save(fig, "ratings_over_time.png")

    def plot_user_activity(self, df: pd.DataFrame):
        user_counts = df.groupby("user_id")["rating"].count()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Log-scale histogram
        axes[0].hist(user_counts, bins=100, color=PALETTE[3], edgecolor="none", log=True)
        axes[0].set_xlabel("Ratings per User")
        axes[0].set_ylabel("Number of Users (log scale)")
        axes[0].set_title("User Activity Distribution")
        axes[0].axvline(user_counts.median(), color="red", linestyle="--",
                        label=f"Median = {user_counts.median():.0f}")
        axes[0].legend()

        # CDF
        sorted_counts = np.sort(user_counts)
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        axes[1].plot(sorted_counts, cdf, color=PALETTE[3], linewidth=2)
        axes[1].set_xlabel("Ratings per User")
        axes[1].set_ylabel("Cumulative Fraction of Users")
        axes[1].set_title("CDF of User Activity")
        axes[1].set_xscale("log")
        axes[1].axhline(0.9, color="gray", linestyle=":", label="90th percentile")
        axes[1].legend()

        fig.suptitle("Netflix Prize — User Activity Patterns", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "user_activity.png")

    def plot_item_popularity(self, df: pd.DataFrame):
        item_counts = df.groupby("item_id")["rating"].count().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Long-tail / Power-law chart
        x = np.arange(1, len(item_counts) + 1)
        axes[0].plot(x, item_counts.values, color=PALETTE[4], linewidth=1.5)
        axes[0].set_xlabel("Movie Rank")
        axes[0].set_ylabel("Number of Ratings")
        axes[0].set_title("Long-Tail Item Popularity Curve")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].fill_between(x, item_counts.values, alpha=0.15, color=PALETTE[4])

        # Top-20 bar chart
        top20 = item_counts.head(20)
        axes[1].barh(range(20), top20.values[::-1], color=PALETTE[0])
        axes[1].set_yticks(range(20))
        axes[1].set_yticklabels([str(i) for i in top20.index[::-1]], fontsize=8)
        axes[1].set_xlabel("Number of Ratings")
        axes[1].set_title("Top-20 Most-Rated Movies (by item_id)")

        fig.suptitle("Netflix Prize — Item Popularity", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "item_popularity.png")

    def plot_sparsity_heatmap(self, df: pd.DataFrame, sample_users: int = 200, sample_items: int = 200):
        """Render a mini user-item matrix heatmap to visualise sparsity."""
        rng = np.random.default_rng(42)
        users = rng.choice(df["user_id"].unique(), size=min(sample_users, df["user_id"].nunique()), replace=False)
        items = rng.choice(df["item_id"].unique(), size=min(sample_items, df["item_id"].nunique()), replace=False)

        sub = df[df["user_id"].isin(users) & df["item_id"].isin(items)]
        matrix = sub.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(12, 8))
        mask = matrix.isna()
        sns.heatmap(
            matrix.fillna(0),
            mask=False,
            cmap="YlOrRd",
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            cbar_kws={"label": "Rating (0 = missing)"},
        )
        sparsity = mask.values.mean()
        ax.set_title(
            f"User-Item Rating Matrix Sample ({sample_users}×{sample_items})\n"
            f"Sparsity = {sparsity:.1%}  |  Black cells = missing ratings",
            fontsize=12,
        )
        ax.set_xlabel("Movies (sampled)")
        ax.set_ylabel("Users (sampled)")
        plt.tight_layout()
        self._save(fig, "sparsity_heatmap.png")

    def plot_rating_by_year(self, df: pd.DataFrame, movies_df: pd.DataFrame = None):
        if "date" not in df.columns:
            return
        df = df.copy()
        df["year"] = df["date"].dt.year
        avg_by_year = df.groupby("year")["rating"].agg(["mean", "count"]).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        ax1.bar(avg_by_year["year"], avg_by_year["count"], alpha=0.4, color=PALETTE[1], label="# Ratings")
        ax2.plot(avg_by_year["year"], avg_by_year["mean"], color=PALETTE[0], marker="o", linewidth=2, label="Avg Rating")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of Ratings")
        ax2.set_ylabel("Average Rating")
        ax2.set_ylim(1, 5)
        ax1.set_title("Rating Volume & Average Rating by Year")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        self._save(fig, "rating_by_year.png")

    def plot_user_item_degree_distribution(self, df: pd.DataFrame):
        """Side-by-side degree distributions for users and items."""
        user_deg = df.groupby("user_id")["item_id"].count()
        item_deg = df.groupby("item_id")["user_id"].count()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, deg, label, color in [
            (axes[0], user_deg, "User Degree (# ratings)", PALETTE[2]),
            (axes[1], item_deg, "Item Degree (# ratings)", PALETTE[3]),
        ]:
            ax.hist(np.log1p(deg), bins=60, color=color, edgecolor="none")
            ax.set_xlabel(f"log(1 + {label})")
            ax.set_ylabel("Count")
            ax.set_title(f"{label} Distribution (log scale)")
            ax.axvline(np.log1p(deg.median()), color="red", linestyle="--",
                       label=f"Median={deg.median():.0f}")
            ax.legend()

        fig.suptitle("Degree Distributions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "degree_distributions.png")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, fname: str):
        path = self.figures_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure → %s", path)
