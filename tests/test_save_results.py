import importlib.util
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script():
    spec = importlib.util.spec_from_file_location("save_results", os.path.join(ROOT, "scripts", "save_results_to_github.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_files_copied_and_large_files_listed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = tmp_path / "drive" / "run_v1"
    (source / "text_eeg").mkdir(parents=True)
    (source / "report.md").write_text("# report")
    (source / "text_eeg" / "fold_0.csv").write_text("a,b\n1,2\n")
    (source / "text_eeg" / "fold_0_weights.pt").write_bytes(b"\0" * 1000)
    (source / "big.csv").write_text("x\n" * 400_000)  # ~0.8 MB, above the limit below
    module = load_script()
    monkeypatch.setattr(sys, "argv", ["x", "--source", str(source), "--name", "demo", "--repo", str(repo),
                                      "--max-mb", "0.5", "--commit", "--author-name", "t", "--author-email", "t@e"])
    module.main()
    saved = repo / "saved_results" / "demo" / "run_v1"
    assert (saved / "report.md").exists() and (saved / "text_eeg" / "fold_0.csv").exists()
    assert not (saved / "text_eeg" / "fold_0_weights.pt").exists() and not (saved / "big.csv").exists()
    listing = (repo / "saved_results" / "demo" / "LARGE_FILES.md").read_text()
    assert "fold_0_weights.pt" in listing and "big.csv" in listing
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True).stdout
    assert "Save results: demo" in log
