from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import fisher_exact

st.title("Enrichment Explorer (ORA)")

root = Path(__file__).resolve().parents[2]
sets_path = root / "data" / "enrichment_sets.csv"
sets_df = pd.read_csv(sets_path)

with st.sidebar:
    st.header("Input gene list")
    mode = st.radio("Source", ["Text", "Upload"], index=0)
    text = st.text_area("Genes (comma or newline)", value="G1\nG2\nG3\nG10\nG20")
    uploaded = st.file_uploader("Upload CSV (column 'gene')", type=["csv"]) 
    top_n = st.slider("Show top N", 5, 50, 15, 1)

if mode == "Text":
    items = [g.strip() for g in text.replace("\n", ",").split(",") if g.strip()]
    input_genes = set(items)
else:
    if uploaded is None:
        st.stop()
    df_in = pd.read_csv(uploaded)
    col = "gene" if "gene" in df_in.columns else df_in.columns[0]
    input_genes = set(df_in[col].astype(str).str.strip().tolist())

universe = set(sets_df["gene"].astype(str).tolist())
input_genes = {g for g in input_genes if g in universe}

def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = pvals * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out

rows = []
for sname, sub in sets_df.groupby("set"):
    s_genes = set(sub["gene"].astype(str))
    a = len(input_genes & s_genes)
    b = len(s_genes - input_genes)
    c = len(input_genes - s_genes)
    d = max(len(universe) - a - b - c, 0)
    table = np.array([[a, b], [c, d]])
    _, p = fisher_exact(table, alternative="greater")
    rows.append((sname, a, len(s_genes), float(p)))

res = pd.DataFrame(rows, columns=["set", "overlap", "set_size", "pvalue"]) 
res["padj"] = bh_adjust(res["pvalue"].to_numpy())
res = res.sort_values(["padj", "overlap"], ascending=[True, False]).head(int(top_n))
st.dataframe(res, use_container_width=True)

if not res.empty:
    plot = res.copy()
    plot["neglog10p"] = -np.log10(np.clip(plot["padj"], 1e-300, None))
    fig = px.bar(plot, x="neglog10p", y="set", orientation="h")
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="-log10(FDR)")
    st.plotly_chart(fig, use_container_width=True)

csv = res.to_csv(index=False).encode()
st.download_button("Download enrichment CSV", data=csv, file_name="enrichment.csv", mime="text/csv")


