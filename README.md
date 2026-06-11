# Netflix Recommendation System

An end-to-end recommendation engine built on the **Netflix Prize dataset** (~100M ratings), implementing two collaborative filtering approaches with full evaluation on RMSE and MAP@10.

## System Architecture

```
Raw Netflix Data (100M ratings)
        │
        ▼
 DataProcessor
  ├── parse_raw_files()
  ├── downsample_data()        ← iterative co-filtering
  └── temporal_split()         ← no leakage

        │
   ┌────┴────┐
 train      test
   │
   ├──► ItemBasedCF             ► MAP@10 (ranking quality)
   │     └── cosine similarity
   │
   └──► MatrixFactorizationSVD  ► RMSE (rating accuracy)
         └── Funk SGD

        │
        ▼
   Evaluator
    ├── RMSE
    └── MAP@10

        │
        ▼
   FastAPI  /api/v1/recommend
    ├── SVD endpoint
    ├── CF endpoint
    └── Cold-start fallback
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/hariom57/Recommendation-System
cd Recommendation-System
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Netflix Prize Dataset

See `data/README.md` for full instructions. Short version:

```bash
pip install kaggle
kaggle datasets download -d netflix-inc/netflix-prize-data
unzip netflix-prize-data.zip -d data/netflix-prize-data/
```

### 3. Train Both Models

```bash
python train_pipeline.py
# Options:
#   --max-rows 5000000   load only 5M rows (default) — fast on a laptop
#   --svd-only           skip ItemBasedCF training
#   --cf-only            skip SVD training
```

This will:
- Parse the raw data and cache it as parquet
- Downsample to a dense sub-matrix
- Perform a temporal train/test split
- Train ItemBasedCF → `models/item_cf.joblib`
- Train MatrixFactorizationSVD → `models/svd.joblib`
- Print RMSE and MAP@10 side-by-side
- Save results to `reports/evaluation_results.csv`

### 4. Run the EDA Notebook

```bash
jupyter notebook notebooks/eda_netflix.ipynb
```

### 5. Start the API

```bash
python app.py
# or: uvicorn app:app --reload
```

API docs at `http://localhost:8000/docs`

Example request:
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "top_k": 10, "model": "svd"}'
```

### 6. Re-evaluate Models (standalone)

```bash
python evaluate_models.py --k 10 --test-sample 100000
```

---

## Model Comparison

| Model | Optimises | Inference cost | Explainability |
|---|---|---|---|
| **MatrixFactorizationSVD** (Funk SVD) | **RMSE** | O(K) | Low |
| **ItemBasedCF** | **MAP@10** | O(K·N_rated) | High |

### MatrixFactorizationSVD
- Learns low-rank user and item embedding matrices via SGD
- Incorporates user bias, item bias, and global mean
- Fast at inference: just a dot product
- Better generalises to sparse users

### ItemBasedCF
- Pre-computes item-item cosine similarity
- Predictions are similarity-weighted averages of rated neighbours
- Highly explainable: "because you liked X, you'll like Y"
- Expensive to compute similarity matrix at scale (O(|I|²))

---

## Project Structure

```
Recommendation-System/
├── app.py                    ← FastAPI server (fixed: lifespan, joblib.load)
├── train_pipeline.py         ← End-to-end training script (NEW)
├── evaluate_models.py        ← Standalone evaluation script (NEW)
├── requirements.txt
├── data/
│   ├── README.md             ← Dataset download instructions (NEW)
│   └── netflix-prize-data/   ← Place raw files here (not committed)
│       ├── combined_data_1.txt
│       ├── ...
│       └── movie_titles.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py    ← Fixed: real Netflix data, temporal split
│   ├── eda.py                ← Fixed: full matplotlib/seaborn plots
│   ├── models.py             ← Fixed: trained models with joblib save/load
│   ├── evaluate.py           ← NEW: RMSE + MAP@10
│   └── explain.py            ← Fixed: real movie name lookup
├── notebooks/
│   └── eda_netflix.ipynb     ← NEW: full EDA with visualisations
├── models/                   ← Saved model files (not committed)
│   ├── svd.joblib
│   └── item_cf.joblib
└── reports/
    ├── evaluation_results.csv
    └── figures/              ← Generated plots
```

---

## Deliverables Checklist

| Requirement | Status |
|---|---|
| Uses Netflix Prize Dataset | ✅ Real data, `DataProcessor.load_data()` |
| EDA with visualisations | ✅ `notebooks/eda_netflix.ipynb` + `src/eda.py` |
| RMSE on real test split | ✅ `evaluate_models.py` + `src/evaluate.py` |
| MAP@10 on real test split | ✅ `evaluate_models.py` + `src/evaluate.py` |
| ≥ 2 models compared | ✅ ItemBasedCF vs MatrixFactorizationSVD |
| Trained models saved to disk | ✅ `joblib.dump()` in `models/` |
| Models loaded in API | ✅ `joblib.load()` in `app.py` lifespan |
| Top-K recommendation demo | ✅ `/api/v1/recommend` endpoint |
| Cold-start handling | ✅ Falls back to popularity-based list |
| Explanation engine | ✅ `src/explain.py` |
| FastAPI deprecation fixed | ✅ Uses `lifespan` context manager |
| GitHub reproducibility | ✅ This README |
