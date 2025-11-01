from pathlib import Path
import streamlit as st

st.set_page_config(page_title="AI Biotech & Translational ML", page_icon="🧬", layout="wide")

root = Path(__file__).resolve().parents[1]
p1 = root / "pfp" / "pfp-maker (5).png"
p2 = root / "@pfp-maker (5).png"
img = p1 if p1.exists() else p2

col_a, col_b = st.columns([1, 3])
with col_a:
    if img.exists():
        st.image(str(img), width=180)
with col_b:
    st.title("Soheil Khosravi")
    st.write("Bioinformatics | Data Analysis for Genomics & Biology | AI Biotech")
    st.write("From raw omics to decisions: clean pipelines, clear visuals, credible results.")

st.subheader("Demos")
st.markdown("""
- Biomarker Classifier: train and evaluate a RandomForest on a sample or uploaded dataset
- Volcano Plot Studio: interactively explore differential expression results
- Drug Response Predictor: regress IC50-like values with RandomForest and SHAP
- scRNA Cluster Viewer: UMAP embedding, k-means clustering, marker scan
- Enrichment Explorer: simple ORA over curated gene sets
- Pathway Highlighter: map DEGs to a stylized pathway diagram and export HTML

Use the sidebar to open each demo.
""")

st.subheader("How I work")
st.markdown("""
✅ I keep communication clear  
✅ My workflows are organized  
✅ Results are aligned with your goals  
✅ I treat every project like a collaboration
""")

st.subheader("Expertise")
st.markdown("""
- Genomics & Transcriptomics: RNA-seq, DNA-seq, single-cell, functional genomics
- Machine Learning & AI for Biotech: biomarker prediction, classification, clustering
- Custom Pipelines: reproducible workflows in Python/R
- Visualization & Reporting: clean, publication-ready figures
""")


