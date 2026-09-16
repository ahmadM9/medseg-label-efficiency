"""The scale launcher keeps its Kaggle defaults and honours the Colab environment."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load(name: str, path: Path, monkeypatch, env: dict[str, str]):
    for var in ("LAUNCH_REPO", "LAUNCH_WORK", "LAUNCH_INPUT", "LAUNCH_MLFLOW"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(name, None)
    spec.loader.exec_module(module)
    return module


def test_scale_launcher_defaults_to_kaggle_paths(monkeypatch):
    m = load("run_scale_default", ROOT / "kaggle/scale/run_scale.py", monkeypatch, {})
    assert m.INPUT == Path("/kaggle/input")
    assert m.WORK == Path("/kaggle/working")
    assert m.REPO_DIR == Path("/tmp/repo")
    assert m.MLFLOW_URI == "sqlite:////kaggle/working/mlflow.db"


def test_scale_launcher_reads_colab_environment(monkeypatch, tmp_path):
    env = {
        "LAUNCH_INPUT": str(tmp_path / "input"),
        "LAUNCH_WORK": str(tmp_path / "drive/work"),
        "LAUNCH_REPO": str(tmp_path / "repo"),
        "LAUNCH_MLFLOW": "sqlite:////content/mlflow.db",
    }
    m = load("run_scale_colab", ROOT / "kaggle/scale/run_scale.py", monkeypatch, env)
    assert m.INPUT == tmp_path / "input"
    assert m.WORK == tmp_path / "drive/work"
    assert m.REPO_DIR == tmp_path / "repo"
    assert m.MLFLOW_URI == "sqlite:////content/mlflow.db"


def test_select_jobs_filters_by_kind_and_rejects_unknown(monkeypatch):
    m = load("run_scale_jobs", ROOT / "kaggle/scale/run_scale.py", monkeypatch, {})
    jobs = [("zeroshot", "sam2", "box"), ("large_ft", "x"), ("incontext", "u", 0, 0)]
    assert m.select_jobs(jobs, "") == jobs
    assert m.select_jobs(jobs, "zeroshot,incontext") == [jobs[0], jobs[2]]
    with pytest.raises(SystemExit):
        m.select_jobs(jobs, "cascade")


def test_prepare_colab_finds_camus_and_stages_checkpoints(monkeypatch, tmp_path):
    m = load("prepare_colab", ROOT / "colab/prepare_colab.py", monkeypatch, {})
    (tmp_path / "input/camus-echo/database_nifti").mkdir(parents=True)
    assert m.find_database_nifti(tmp_path / "input").parent.name == "camus-echo"
    src = tmp_path / "drive/checkpoints/outputs/camus_unet2d_p05"
    src.mkdir(parents=True)
    (src / "best.pt").write_bytes(b"x")
    dst = m.stage_checkpoints(tmp_path / "drive", tmp_path / "input")
    assert (dst / "camus_unet2d_p05/best.pt").exists()
    # a second call is a no-op, the copy is not repeated
    assert m.stage_checkpoints(tmp_path / "drive", tmp_path / "input") == dst
    assert m.stage_checkpoints(tmp_path / "empty-drive", tmp_path / "input") is None


def test_prepare_colab_requires_both_secrets(monkeypatch):
    m = load("prepare_colab_env", ROOT / "colab/prepare_colab.py", monkeypatch, {})
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(SystemExit):
        m.kaggle_credentials()
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert m.kaggle_credentials() == {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
