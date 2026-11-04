import joblib
import numpy as np
import pandas as pd
import pytest

from automl.artifacts import RunArtifacts
from automl.exceptions import ModelArtifactError


def test_csv_round_trip(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="csv-run")
    original = pd.DataFrame({"number": [1, 2], "label": ["a", "b"]})

    reference = artifacts.write_csv("input/data.csv", original)
    loaded = RunArtifacts.open(tmp_path / "artifacts", "csv-run").read_csv(reference)

    pd.testing.assert_frame_equal(loaded, original)


def test_json_round_trip(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="json-run")
    original = {"metrics": {"accuracy": 0.9}, "warnings": ["sample warning"]}

    reference = artifacts.write_json("reports/validation.json", original)

    assert artifacts.read_json(reference) == original


def test_joblib_round_trip(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="joblib-run")
    original = {"weights": np.array([1.5, 2.5]), "labels": ("a", "b")}

    reference = artifacts.write_joblib("stages/prepared.joblib", original)
    loaded = artifacts.read_joblib(reference)

    np.testing.assert_array_equal(loaded["weights"], original["weights"])
    assert loaded["labels"] == original["labels"]


def test_generated_run_directories_are_unique(tmp_path):
    artifact_root = tmp_path / "artifacts"

    first = RunArtifacts.create(artifact_root)
    second = RunArtifacts.create(artifact_root)

    assert first.run_id != second.run_id
    assert first.directory.is_dir()
    assert second.directory.is_dir()


def test_empty_run_id_is_rejected(tmp_path):
    with pytest.raises(ModelArtifactError, match="Invalid artifact run ID"):
        RunArtifacts.create(tmp_path / "artifacts", run_id="")


def test_existing_run_id_is_not_reused_or_cleared(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="fixed-run")
    artifacts.write_json("keep.json", {"keep": True})

    with pytest.raises(ModelArtifactError, match="Could not create"):
        RunArtifacts.create(tmp_path / "artifacts", run_id="fixed-run")

    assert artifacts.read_json("keep.json") == {"keep": True}


@pytest.mark.parametrize(
    "reference",
    [
        "../outside.json",
        "nested/../../outside.json",
        "/tmp/outside.json",
        r"C:\outside.json",
        "",
        ".",
    ],
)
def test_invalid_artifact_references_are_rejected(tmp_path, reference):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="safe-run")

    with pytest.raises(ModelArtifactError, match="Invalid artifact reference"):
        artifacts.write_json(reference, {"unsafe": True})


def test_symlink_reference_cannot_escape_run_directory(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="symlink-run")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = artifacts.directory / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable in this environment")

    with pytest.raises(ModelArtifactError, match="escapes run directory"):
        artifacts.write_json("linked/escape.json", {"unsafe": True})

    assert not (outside / "escape.json").exists()


def test_open_rejects_invalid_or_missing_run_directory(tmp_path):
    with pytest.raises(ModelArtifactError, match="Invalid artifact run ID"):
        RunArtifacts.open(tmp_path, "../outside")

    with pytest.raises(ModelArtifactError, match="does not exist"):
        RunArtifacts.open(tmp_path, "missing-run")


def test_malformed_json_raises_artifact_error(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="bad-json")
    target = artifacts.directory / "broken.json"
    target.write_text('{"unfinished": ', encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="Could not read JSON"):
        artifacts.read_json("broken.json")


def test_failed_json_serialization_preserves_previous_final_file(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="json-failure")
    artifacts.write_json("result.json", {"status": "old"})

    with pytest.raises(ModelArtifactError, match="Could not write artifact"):
        artifacts.write_json("result.json", {"unsupported": object()})

    assert artifacts.read_json("result.json") == {"status": "old"}
    assert list(artifacts.directory.glob(".result.json.*.tmp")) == []


def test_failed_joblib_write_preserves_previous_final_file(tmp_path, monkeypatch):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="joblib-failure")
    artifacts.write_joblib("result.joblib", {"status": "old"})

    def fail_after_partial_write(value, filename, *args, **kwargs):
        with open(filename, "wb") as file:
            file.write(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(joblib, "dump", fail_after_partial_write)
    with pytest.raises(ModelArtifactError, match="Could not write artifact"):
        artifacts.write_joblib("result.joblib", {"status": "new"})

    assert artifacts.read_joblib("result.joblib") == {"status": "old"}
    assert list(artifacts.directory.glob(".result.joblib.*.tmp")) == []


def test_json_is_strict_about_non_finite_numbers(tmp_path):
    artifacts = RunArtifacts.create(tmp_path / "artifacts", run_id="strict-json")

    with pytest.raises(ModelArtifactError, match="Could not write artifact"):
        artifacts.write_json("metrics.json", {"score": float("nan")})

    assert not (artifacts.directory / "metrics.json").exists()
