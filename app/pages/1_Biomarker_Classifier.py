from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

st.title("Biomarker Classifier")

with st.sidebar:
    st.header("Data")
    data_mode = st.radio("Source", ["BreastCancer", "Synthetic", "Upload"], index=0)
    target_col = st.text_input("Target column (for upload)", value="target")
    st.header("Model")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    n_estimators = st.slider("n_estimators", 50, 800, 300, 50)
    max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20], index=0)
    st.header("Explainers")
    show_shap = st.checkbox("Compute SHAP", value=True)

def load_sample(n_samples: int, n_features: int, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(1, n_features // 10),
        weights=[0.6, 0.4],
        random_state=random_state,
    )
    cols = [f"feat_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    s = pd.Series(y, name="target")
    return df, s

def load_upload(uploaded, target: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    if uploaded is None:
        return None
    df = pd.read_csv(uploaded)
    if target not in df.columns:
        st.error(f"Target column '{target}' not found")
        return None
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

if data_mode == "BreastCancer":
    ds = load_breast_cancer(as_frame=True)
    X, y = ds.data, ds.target
elif data_mode == "Synthetic":
    X, y = load_sample(600, 30, random_state)
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    loaded = load_upload(uploaded, target_col)
    if loaded is None:
        st.stop()
    X, y = loaded

md = None if max_depth_opt == "None" else int(max_depth_opt)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=md, random_state=random_state)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
proba_cv = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_test, proba)
pr_auc = auc(rec, prec)
roc_cv = roc_auc_score(y, proba_cv)
pr_cv = average_precision_score(y, proba_cv)
cm = confusion_matrix(y_test, pred)

col1, col2 = st.columns(2)
with col1:
    st.metric("ROC AUC (test)", f"{roc_auc:.3f}")
with col2:
    st.metric("PR AUC (test)", f"{pr_auc:.3f}")

col3, col4 = st.columns(2)
with col3:
    st.metric("ROC AUC (CV)", f"{roc_cv:.3f}")
with col4:
    st.metric("PR AUC (CV)", f"{pr_cv:.3f}")

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.3f})"))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash")))
roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white")
st.plotly_chart(roc_fig, use_container_width=True)

pr_fig = go.Figure()
pr_fig.add_trace(go.Scatter(x=rec, y=prec, name=f"PR (AUC={pr_auc:.3f})"))
pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", template="plotly_white")
st.plotly_chart(pr_fig, use_container_width=True)

cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
cm_fig.update_xaxes(title="Predicted")
cm_fig.update_yaxes(title="True")
st.plotly_chart(cm_fig, use_container_width=True)

imp = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_}).sort_values("importance", ascending=False).head(20)
imp_fig = px.bar(imp, x="importance", y="feature", orientation="h")
imp_fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(imp_fig, use_container_width=True)

pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": pred, "y_proba": proba})
csv = pred_df.to_csv(index=False).encode()
st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

if show_shap:
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        vals = np.abs(sv).mean(0)
        shap_imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False).head(20)
        shap_fig = px.bar(shap_imp, x="mean_abs_shap", y="feature", orientation="h")
        shap_fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(shap_fig, use_container_width=True)
    except Exception:
        st.info("Install 'shap' to enable SHAP explainability: pip install shap")


