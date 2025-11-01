import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.title("Drug Response Predictor")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Synthetic", "Upload"], index=0)
    target_col = st.text_input("Target column (for upload)", value="ic50")
    st.header("Model")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    n_estimators = st.slider("n_estimators", 50, 800, 300, 50)
    max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20], index=0)
    show_shap = st.checkbox("Compute SHAP", value=True)

def load_synth(n_samples: int, n_features: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
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

def load_upload(uploaded, target: str):
    if uploaded is None:
        return None
    df = pd.read_csv(uploaded)
    if target not in df.columns:
        st.error(f"Target column '{target}' not found")
        return None
    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])
    X = X.apply(pd.to_numeric, errors="coerce")
    m = pd.concat([X, y], axis=1).dropna()
    y = m[target]
    X = m.drop(columns=[target])
    return X, y

if mode == "Synthetic":
    X, y = load_synth(600, 30, random_state)
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    loaded = load_upload(uploaded, target_col)
    if loaded is None:
        st.stop()
    X, y = loaded

md = None if max_depth_opt == "None" else int(max_depth_opt)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=md, random_state=random_state)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R2", f"{r2:.3f}")
with col2:
    st.metric("MAE", f"{mae:.3f}")
with col3:
    st.metric("RMSE", f"{rmse:.3f}")

sc = px.scatter(x=y_test, y=y_pred, labels={"x": "True IC50", "y": "Predicted IC50"})
sc.add_trace(go.Scatter(x=[float(y_test.min()), float(y_test.max())], y=[float(y_test.min()), float(y_test.max())], mode="lines", name="Ideal", line=dict(dash="dash")))
st.plotly_chart(sc, use_container_width=True)

resid = y_test - y_pred
hist = px.histogram(resid, nbins=30)
hist.update_layout(xaxis_title="Residuals (True - Pred)", yaxis_title="Count")
st.plotly_chart(hist, use_container_width=True)

imp = pd.DataFrame({"feature": X.columns, "importance": reg.feature_importances_}).sort_values("importance", ascending=False).head(20)
imp_fig = px.bar(imp, x="importance", y="feature", orientation="h")
imp_fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(imp_fig, use_container_width=True)

if show_shap:
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(reg)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        vals = np.abs(sv).mean(0)
        shap_imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False).head(20)
        shap_fig = px.bar(shap_imp, x="mean_abs_shap", y="feature", orientation="h")
        shap_fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(shap_fig, use_container_width=True)
    except Exception:
        st.info("Install 'shap' to enable SHAP explainability: pip install shap")

pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
csv = pred_df.to_csv(index=False).encode()
st.download_button("Download predictions CSV", data=csv, file_name="drug_predictions.csv", mime="text/csv")


