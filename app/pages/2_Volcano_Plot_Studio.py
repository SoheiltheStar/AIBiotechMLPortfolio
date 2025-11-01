from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Volcano Plot Studio")

mode = st.radio("Source", ["Sample", "Upload"], index=0)
uploaded = None
if mode == "Upload":
    uploaded = st.file_uploader("Upload CSV (gene,log2FC,pvalue)", type=["csv"]) 

if mode == "Sample":
    root = Path(__file__).resolve().parents[2]
    fp = root / "data" / "degs_sample.csv"
    df = pd.read_csv(fp)
else:
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)

df = df.copy()
df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
df = df.dropna(subset=["log2FC", "pvalue"]) 

col1, col2, col3 = st.columns(3)
with col1:
    fc_thr = st.slider("|log2FC| threshold", 0.5, 3.0, 1.0, 0.05)
with col2:
    pv_thr = st.number_input("p-value threshold", min_value=1e-6, max_value=0.5, value=0.05, step=0.0001, format="%f")
with col3:
    top_k = st.number_input("Top labels", min_value=0, max_value=30, value=10, step=1)

df["status"] = "NS"
df.loc[(df["log2FC"] >= fc_thr) & (df["pvalue"] <= pv_thr), "status"] = "Up"
df.loc[(df["log2FC"] <= -fc_thr) & (df["pvalue"] <= pv_thr), "status"] = "Down"
df["neglog10p"] = -np.log10(np.clip(df["pvalue"], 1e-300, None))

sig = df[df["status"] != "NS"].sort_values(["pvalue", "log2FC"], ascending=[True, False]).head(int(top_k))
df["label"] = ""
df.loc[sig.index, "label"] = sig["gene"].astype(str)

palette = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#7f7f7f"}
fig = px.scatter(df, x="log2FC", y="neglog10p", color="status", color_discrete_map=palette, hover_name="gene", opacity=0.8, text="label")
fig.update_traces(textposition="top center")
fig.add_hline(y=-np.log10(pv_thr), line_dash="dash")
fig.add_vline(x=fc_thr, line_dash="dash")
fig.add_vline(x=-fc_thr, line_dash="dash")
fig.update_layout(xaxis_title="log2 fold-change", yaxis_title="-log10(p-value)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

counts = df["status"].value_counts().reindex(["Up", "Down", "NS"]).fillna(0).astype(int)
st.write({"Up": int(counts.get("Up", 0)), "Down": int(counts.get("Down", 0)), "NS": int(counts.get("NS", 0))})

out = df[["gene", "log2FC", "pvalue", "status"]]
csv = out.to_csv(index=False).encode()
st.download_button("Download table CSV", data=csv, file_name="volcano_table.csv", mime="text/csv")


