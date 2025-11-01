from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import joblib


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name == "breast_cancer":
        ds = load_breast_cancer(as_frame=True)
        X = ds.data
        y = ds.target
        return X, y
    raise ValueError("unsupported dataset")


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    if metric == "f1":
        best_t = 0.5
        best = -1.0
        for t in np.linspace(0.05, 0.95, 19):
            pred = (y_proba >= t).astype(int)
            s = f1_score(y_true, pred)
            if s > best:
                best = s
                best_t = t
        return float(best_t)
    raise ValueError("unsupported metric")


def train_and_save(dataset: str, n_estimators: int, max_depth: int | None, seed: int, out_dir: Path) -> None:
    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    clf.fit(X_train, y_train)
    proba_test = clf.predict_proba(X_test)[:, 1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba_cv = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    roc = roc_auc_score(y_test, proba_test)
    pr = average_precision_score(y_test, proba_test)
    t_opt = optimize_threshold(y_test.to_numpy(), proba_test)
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    metrics = {
        "dataset": dataset,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "seed": seed,
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold_opt_f1": float(t_opt),
        "precision_points": prec.tolist()[:200],
        "recall_points": rec.tolist()[:200],
        "feature_names": list(X.columns),
        "feature_importances": clf.feature_importances_.tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breast_cancer")
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="artifacts")
    a = p.parse_args()
    md = None if a.max_depth == 0 else a.max_depth
    train_and_save(a.dataset, a.n_estimators, md, a.seed, Path(a.out_dir))


if __name__ == "__main__":
    main()


