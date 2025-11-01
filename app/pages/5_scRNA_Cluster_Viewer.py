from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

st.title("scRNA Cluster Viewer")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Synthetic", "Upload"], index=0)
    n_cells = st.slider("Cells (synthetic)", 200, 5000, 800, 100)
    n_genes = st.slider("Genes (synthetic)", 20, 200, 50, 5)
    n_classes = st.slider("Clusters (synthetic)", 2, 10, 4, 1)
    target_col = st.text_input("Target column (optional)", value="")
    st.header("Embedding & Clustering")
    n_neighbors = st.slider("n_neighbors", 5, 50, 15, 1)
    min_dist = st.slider("min_dist", 0.0, 0.9, 0.1, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    k_clusters = st.slider("k-means k", 2, 12, 4, 1)
    top_markers = st.slider("Top markers per cluster", 3, 30, 10, 1)

def load_synth(n_samples: int, n_features: int, n_classes: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(4, n_features // 2),
        n_redundant=max(2, n_features // 10),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    genes = [f"G{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=genes)
    s = pd.Series(y, name="label")
    return df, s

def load_upload(uploaded, target: str):
    if uploaded is None:
        return None
    df = pd.read_csv(uploaded)
    if target and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")
    return X, y

if mode == "Synthetic":
    X, y = load_synth(n_cells, n_genes, n_classes, random_state)
else:
    uploaded = st.file_uploader("Upload CSV (cells x genes)", type=["csv"]) 
    loaded = load_upload(uploaded, target_col)
    if loaded is None:
        st.stop()
    X, y = loaded

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
um = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
emb = um.fit_transform(Xs)
km = KMeans(n_clusters=k_clusters, n_init=10, random_state=random_state)
clusters = km.fit_predict(Xs)

df_plot = pd.DataFrame({"UMAP1": emb[:, 0], "UMAP2": emb[:, 1], "cluster": clusters.astype(int)})
if y is not None:
    df_plot["label"] = y.reset_index(drop=True)

color_by = st.selectbox("Color by", ["cluster"] + (["label"] if y is not None else []))
fig = px.scatter(df_plot, x="UMAP1", y="UMAP2", color=color_by, opacity=0.85)
st.plotly_chart(fig, use_container_width=True)

gene_select = st.selectbox("Gene expression overlay", [None] + list(X.columns))
if gene_select is not None:
    df_exp = df_plot.copy()
    df_exp["expr"] = X[gene_select].to_numpy()
    fig2 = px.scatter(df_exp, x="UMAP1", y="UMAP2", color="expr", color_continuous_scale="Viridis")
    st.plotly_chart(fig2, use_container_width=True)

X_np = X.to_numpy()
markers = []
for c in np.unique(clusters):
    idx = clusters == c
    mean_in = X_np[idx].mean(axis=0)
    mean_out = X_np[~idx].mean(axis=0)
    score = mean_in - mean_out
    top_idx = np.argsort(score)[::-1][:int(top_markers)]
    for gi in top_idx:
        markers.append((int(c), X.columns[int(gi)], float(score[int(gi)])))
marker_df = pd.DataFrame(markers, columns=["cluster", "gene", "score"]).sort_values(["cluster", "score"], ascending=[True, False])
st.dataframe(marker_df, use_container_width=True)
csv = marker_df.to_csv(index=False).encode()
st.download_button("Download markers CSV", data=csv, file_name="scrna_markers.csv", mime="text/csv")


