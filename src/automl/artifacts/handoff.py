"""Safe, atomic file handoffs scoped to a unique pipeline run directory.

Artifact references are paths relative to one run directory. The class supports
reopening a run from another process, so callers can exchange references rather
than DataFrames, fitted transformers, or models through Airflow XCom.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import joblib
import pandas as pd

from automl.exceptions import ModelArtifactError


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RunArtifacts:
    """Read and atomically write artifacts inside one run directory.

    Use :meth:`create` to create a unique run, or :meth:`open` to access an
    existing run from another process. All artifact references must be relative
    to the run directory and may not escape it.
    """

    root: Path
    run_id: str

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def directory(self) -> Path:
        """Absolute path to this run's directory."""

        return self.root / self.run_id

    @classmethod
    def create(
        cls,
        artifact_root: str | Path,
        run_id: str | None = None,
    ) -> RunArtifacts:
        """Create a new run directory under ``artifact_root``.

        When ``run_id`` is omitted, a UUID hex string is generated. Existing
        run directories are never reused or cleared.
        """

        selected_run_id = uuid.uuid4().hex if run_id is None else run_id
        _validate_run_id(selected_run_id)
        root = Path(artifact_root).expanduser().resolve()
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / selected_run_id).mkdir()
        except OSError as exc:
            raise ModelArtifactError(
                f"Could not create artifact run directory '{selected_run_id}' under {root}"
            ) from exc
        return cls(root=root, run_id=selected_run_id)

    @classmethod
    def open(cls, artifact_root: str | Path, run_id: str) -> RunArtifacts:
        """Open an existing run directory without creating or modifying it."""

        _validate_run_id(run_id)
        artifacts = cls(root=Path(artifact_root), run_id=run_id)
        try:
            if not artifacts.directory.is_dir():
                raise ModelArtifactError(
                    f"Artifact run directory does not exist: {artifacts.directory}"
                )
            if artifacts.directory.resolve().parent != artifacts.root:
                raise ModelArtifactError("Artifact run directory escapes the artifact root")
        except OSError as exc:
            raise ModelArtifactError(
                f"Could not access artifact run directory: {artifacts.directory}"
            ) from exc
        return artifacts

    def write_csv(self, reference: str | Path, data_frame: pd.DataFrame) -> str:
        """Atomically write a DataFrame as CSV and return its relative reference."""

        target, relative = self._target(reference)

        def write(temporary_path: Path) -> None:
            data_frame.to_csv(temporary_path, index=False)

        self._atomic_write(target, write)
        return relative

    def read_csv(self, reference: str | Path) -> pd.DataFrame:
        """Read a CSV artifact from this run."""

        target, _ = self._target(reference)
        self._require_file(target)
        try:
            return pd.read_csv(target)
        except Exception as exc:
            raise ModelArtifactError(f"Could not read CSV artifact '{reference}'") from exc

    def write_json(self, reference: str | Path, value: Any) -> str:
        """Atomically write a JSON-serializable value and return its reference."""

        target, relative = self._target(reference)

        def write(temporary_path: Path) -> None:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
                file.write("\n")

        self._atomic_write(target, write)
        return relative

    def read_json(self, reference: str | Path) -> Any:
        """Read JSON, raising :class:`ModelArtifactError` for malformed files."""

        target, _ = self._target(reference)
        self._require_file(target)
        try:
            with target.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Could not read JSON artifact '{reference}'") from exc

    def write_joblib(self, reference: str | Path, value: Any) -> str:
        """Atomically serialize a Python object with joblib and return its reference."""

        target, relative = self._target(reference)

        def write(temporary_path: Path) -> None:
            joblib.dump(value, temporary_path)

        self._atomic_write(target, write)
        return relative

    def read_joblib(self, reference: str | Path) -> Any:
        """Load a joblib artifact from this run."""

        target, _ = self._target(reference)
        self._require_file(target)
        try:
            return joblib.load(target)
        except Exception as exc:
            raise ModelArtifactError(f"Could not load joblib artifact '{reference}'") from exc

    def _target(self, reference: str | Path) -> tuple[Path, str]:
        """Resolve and validate a run-relative artifact reference."""

        raw_reference = str(reference)
        candidate = Path(raw_reference)
        windows_candidate = PureWindowsPath(raw_reference)
        if (
            not raw_reference
            or candidate.is_absolute()
            or windows_candidate.is_absolute()
            or windows_candidate.root
            or windows_candidate.drive
            or ".." in candidate.parts
            or ".." in windows_candidate.parts
            or candidate.name in {"", "."}
        ):
            raise ModelArtifactError(f"Invalid artifact reference: {reference!s}")

        target = self.directory / candidate
        try:
            resolved_run_directory = self.directory.resolve()
            if resolved_run_directory.parent != self.root:
                raise ModelArtifactError(
                    "Artifact run directory escapes the configured artifact root"
                )
            resolved_target = target.resolve()
            resolved_target.relative_to(resolved_run_directory)
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(
                f"Artifact reference escapes run directory: {reference!s}"
            ) from exc

        # Return a normalized POSIX reference for metadata and XCom use.
        return resolved_target, candidate.as_posix()

    def _atomic_write(self, target: Path, writer: Any) -> None:
        """Write to a sibling temporary file and atomically replace the target."""

        temporary_path: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recheck after creating parents, since an existing symlink must not
            # redirect a write outside this run directory.
            target, _ = self._target(target.relative_to(self.directory))
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            writer(temporary_path)
            os.replace(temporary_path, target)
            temporary_path = None
        except ModelArtifactError:
            raise
        except Exception as exc:
            raise ModelArtifactError(f"Could not write artifact '{target.name}'") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _require_file(target: Path) -> None:
        try:
            if not target.is_file():
                raise ModelArtifactError(f"Artifact file does not exist: {target}")
        except OSError as exc:
            raise ModelArtifactError(f"Could not access artifact file: {target}") from exc


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or not _RUN_ID_PATTERN.fullmatch(run_id)
        or run_id in {".", ".."}
    ):
        raise ModelArtifactError(f"Invalid artifact run ID: {run_id!r}")
