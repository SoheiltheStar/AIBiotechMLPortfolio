from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.title("Pathway Highlighter (Stylized KEGG)")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Sample", "Upload"], index=0)
    fc_thr = st.slider("|log2FC| threshold", 0.5, 3.0, 1.0, 0.05)
    p_thr = st.number_input("p-value threshold", min_value=1e-6, max_value=0.5, value=0.05, step=0.0001, format="%f")
    top_n = st.slider("Top nodes", 5, 40, 15, 1)
    layers = 3

def load_data_sample() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    fp = root / "data" / "degs_sample.csv"
    return pd.read_csv(fp)

def load_data_upload(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    df = pd.read_csv(uploaded)
    need = {"gene", "log2FC", "pvalue"}
    if not need.issubset(set(df.columns)):
        st.error("CSV must have columns: gene, log2FC, pvalue")
        return None
    return df

if mode == "Sample":
    df = load_data_sample()
else:
    up = st.file_uploader("Upload CSV (gene,log2FC,pvalue)", type=["csv"]) 
    loaded = load_data_upload(up)
    if loaded is None:
        st.stop()
    df = loaded

df = df.copy()
df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
df = df.dropna(subset=["log2FC", "pvalue"]) 
df["status"] = "NS"
df.loc[(df["log2FC"] >= fc_thr) & (df["pvalue"] <= p_thr), "status"] = "Up"
df.loc[(df["log2FC"] <= -fc_thr) & (df["pvalue"] <= p_thr), "status"] = "Down"
df["neglog10p"] = -np.log10(np.clip(df["pvalue"], 1e-300, None))

sig = df[df["status"] != "NS"].sort_values(["pvalue", "log2FC"], ascending=[True, False]).head(int(top_n))
if sig.empty:
    st.warning("No significant genes under current thresholds.")
    st.stop()

def layout_nodes(genes: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    n = len(genes)
    per_layer = int(np.ceil(n / n_layers))
    xs = []
    ys = []
    layer_ids = []
    for li in range(n_layers):
        start = li * per_layer
        end = min((li + 1) * per_layer, n)
        count = max(end - start, 0)
        if count <= 0:
            continue
        x = li / (n_layers - 1) if n_layers > 1 else 0.5
        y_positions = np.linspace(0.1, 0.9, count)
        xs.extend([x] * count)
        ys.extend(list(y_positions))
        layer_ids.extend([li] * count)
    out = genes.copy().reset_index(drop=True)
    out["x"] = xs
    out["y"] = ys
    out["layer"] = layer_ids
    return out

nodes = layout_nodes(sig[["gene", "log2FC", "pvalue", "status", "neglog10p"]], layers)

edges_x = []
edges_y = []
for li in range(layers - 1):
    a = nodes[nodes["layer"] == li]
    b = nodes[nodes["layer"] == (li + 1)]
    m = min(len(a), len(b))
    if m == 0:
        continue
    a_idx = a.index[:m]
    b_idx = b.index[:m]
    for i in range(m):
        edges_x.extend([nodes.loc[a_idx[i], "x"], nodes.loc[b_idx[i], "x"], None])
        edges_y.extend([nodes.loc[a_idx[i], "y"], nodes.loc[b_idx[i], "y"], None])

palette = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#7f7f7f"}
size = 12 + (nodes["neglog10p"] / max(nodes["neglog10p"].max(), 1.0) * 18.0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines", line=dict(color="#cccccc", width=1), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(
    x=nodes["x"], y=nodes["y"], mode="markers+text", text=nodes["gene"], textposition="top center",
    marker=dict(size=size, color=[palette.get(s, "#7f7f7f") for s in nodes["status"]], line=dict(width=1, color="#333")),
    hovertemplate="gene=%{text}<br>log2FC=%{customdata[0]:.2f}<br>p=%{customdata[1]:.2e}<extra></extra>",
    customdata=np.stack([nodes["log2FC"].to_numpy(), nodes["pvalue"].to_numpy()], axis=1),
    showlegend=False,
))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Nodes")
st.dataframe(nodes[["gene", "log2FC", "pvalue", "status"]], use_container_width=True)

html = fig.to_html(full_html=False, include_plotlyjs="cdn")
st.download_button("Download figure HTML", data=html.encode(), file_name="pathway_highlighter.html", mime="text/html")


