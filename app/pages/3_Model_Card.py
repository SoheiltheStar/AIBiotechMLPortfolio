from pathlib import Path
import json
import plotly.express as px
import pandas as pd
import streamlit as st

st.title("Model Card")

root = Path(__file__).resolve().parents[2]
metrics_path = root / "artifacts" / "metrics.json"
if not metrics_path.exists():
    st.warning("No artifacts found. Run the training script first.")
    st.stop()

with open(metrics_path, "r", encoding="utf-8") as f:
    m = json.load(f)

st.subheader("Summary")
st.write({
    "dataset": m.get("dataset"),
    "roc_auc": round(m.get("roc_auc", 0.0), 3),
    "pr_auc": round(m.get("pr_auc", 0.0), 3),
    "threshold_opt_f1": round(m.get("threshold_opt_f1", 0.5), 3),
    "n_estimators": m.get("n_estimators"),
    "max_depth": m.get("max_depth"),
})

imp = pd.DataFrame({
    "feature": m.get("feature_names", []),
    "importance": m.get("feature_importances", []),
}).sort_values("importance", ascending=False).head(20)
fig = px.bar(imp, x="importance", y="feature", orientation="h")
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)


