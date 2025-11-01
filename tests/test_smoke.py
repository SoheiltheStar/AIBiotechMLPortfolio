from pathlib import Path
import pandas as pd


def test_data_sample_exists_and_valid():
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "degs_sample.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert {"gene", "log2FC", "pvalue"}.issubset(set(df.columns))


def test_pages_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app" / "Home.py").exists()
    assert (root / "app" / "pages" / "1_Biomarker_Classifier.py").exists()
    assert (root / "app" / "pages" / "2_Volcano_Plot_Studio.py").exists()
    assert (root / "app" / "pages" / "3_Model_Card.py").exists()
    assert (root / "app" / "pages" / "4_Drug_Response_Predictor.py").exists()
    assert (root / "app" / "pages" / "5_scRNA_Cluster_Viewer.py").exists()
    assert (root / "app" / "pages" / "6_Enrichment_Explorer.py").exists()
    assert (root / "app" / "pages" / "7_Pathway_Highlighter.py").exists()


