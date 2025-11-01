from pathlib import Path
import json
import subprocess
import sys

import httpx


def test_training_script_drug_produces_artifacts(tmp_path: Path):
    proj = Path(__file__).resolve().parents[1]
    out = tmp_path / "artifacts_drug"
    cmd = [sys.executable, str(proj / "scripts" / "train_drug.py"), "--out_dir", str(out)]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert (out / "model.joblib").exists()
    assert (out / "metrics.json").exists()
    m = json.loads((out / "metrics.json").read_text())
    assert "r2" in m and m["r2"] > 0.6


def test_api_drug_predict(monkeypatch, tmp_path: Path):
    proj = Path(__file__).resolve().parents[1]
    art = tmp_path / "artifacts_drug"
    subprocess.run([sys.executable, str(proj / "scripts" / "train_drug.py"), "--out_dir", str(art)], check=True)
    code = (proj / "service" / "api_drug.py").read_text()
    code = code.replace("root = Path(__file__).resolve().parents[1]", f"root = Path(r'{tmp_path}')")
    p = tmp_path / "api_drug_tmp.py"
    p.write_text(code)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", f"{p.stem}:app", "--port", "8012"], cwd=str(tmp_path))
    try:
        with httpx.Client(timeout=10) as c:
            for _ in range(30):
                try:
                    r = c.get("http://127.0.0.1:8012/health")
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
            r = c.post("http://127.0.0.1:8012/predict", json={"features": [0.0] * 30})
            assert r.status_code in (200, 400)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


