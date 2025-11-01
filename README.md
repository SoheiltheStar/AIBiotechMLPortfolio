# AI Biotech & Translational ML — Portfolio Demos

Expert-focused Streamlit demos with training pipeline, model card, explainability, and an optional FastAPI service.

## Features

- Biomarker Classifier: RandomForest with train/test split, 5-fold CV metrics, calibration view, threshold optimization, SHAP feature explainability
- Volcano Plot Studio: interactive thresholds, labels, counts, and CSV export
- Training Pipeline: `scripts/train.py` saves `artifacts/model.joblib` and `artifacts/metrics.json`
- Model Card: renders metrics and top features from artifacts
- FastAPI Service: `/health`, `/predict`, `/predict_batch` loading saved model
- Drug Response Predictor: regression with RandomForest, SHAP, residual analysis
  - Training: `scripts/train_drug.py` → `artifacts_drug/`
  - API: `uvicorn service.api_drug:app --port 8012`
- scRNA Cluster Viewer: UMAP embedding, k-means clustering, marker scan, gene overlay
- Enrichment Explorer: simple ORA with Fisher test over curated sets
- Pathway Highlighter: map DEGs to a stylized pathway diagram; export as HTML

## Quickstart

1. Create and activate a virtual environment
2. Install: `pip install -e .[dev]`
3. Train and save artifacts: `python scripts/train.py --out_dir artifacts`
4. Run app: `streamlit run app/Home.py`
5. Run API (optional): `uvicorn service.api:app --port 8010`
6. Tests: `pytest -q`

## Data formats

- Volcano CSV: `gene,log2FC,pvalue`
- Classifier upload CSV: features + one binary `target` column

## Pages

- `Home` overview
- `Biomarker Classifier` with CV, calibration, SHAP
- `Volcano Plot Studio`
- `Model Card` from saved artifacts

