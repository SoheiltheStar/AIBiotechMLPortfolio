from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


def load_dataset(n_samples: int, n_features: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        noise=10.0,
        random_state=seed,
    )
    y = y - y.min()
    y = y / (y.max() + 1e-9)
    y = y * 8.0 + 2.0
    cols = [f"feat_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    s = pd.Series(y, name="ic50")
    return df, s


def train_and_save(n_samples: int, n_features: int, n_estimators: int, max_depth: int | None, seed: int, out_dir: Path) -> None:
    X, y = load_dataset(n_samples, n_features, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_cv = cross_val_predict(reg, X, y, cv=kf)
    r2_cv = r2_score(y, y_cv)
    rmse_cv = mean_squared_error(y, y_cv, squared=False)
    metrics = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "seed": int(seed),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_cv": float(r2_cv),
        "rmse_cv": float(rmse_cv),
        "feature_names": list(X.columns),
        "feature_importances": reg.feature_importances_.tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--n_features", type=int, default=30)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="artifacts_drug")
    a = p.parse_args()
    md = None if a.max_depth == 0 else a.max_depth
    train_and_save(a.n_samples, a.n_features, a.n_estimators, md, a.seed, Path(a.out_dir))


if __name__ == "__main__":
    main()


