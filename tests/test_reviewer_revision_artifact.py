import json
import subprocess
import sys
import zipfile

import pytest

import scripts.build_reviewer_revision_artifact as artifact_builder
from scripts.build_reviewer_revision_artifact import (
    EVIDENCE_ROOTS,
    ROOT,
    include_in_compact,
    inventory_identity,
    sha256_file,
    verify_zip_archive,
    write_deterministic_zip,
)


def test_artifact_paths_and_hashes_are_portable(tmp_path):
    source = tmp_path / "row.json"
    source.write_text('{"ok":true}\n', encoding="utf-8")
    assert include_in_compact(source)
    assert sha256_file(source) == "7d167dac6a0d1dcd20ac46505b63886dd6a8a972fc222a9c7eead0187976895c"
    assert ROOT.is_absolute()


def test_large_prediction_is_indexed_instead_of_duplicated(tmp_path):
    source = tmp_path / "prediction.npz"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    assert not include_in_compact(source)


def test_reviewer_revision_results_are_included():
    assert "reports/v7/reviewer_revision" in EVIDENCE_ROOTS


def test_inventory_identity_is_order_independent_but_path_sensitive():
    first = {
        "b.txt": {"sha256": "2", "bytes": 2},
        "a.txt": {"sha256": "1", "bytes": 1},
    }
    second = {
        "a.txt": {"sha256": "1", "bytes": 1},
        "b.txt": {"sha256": "2", "bytes": 2},
    }
    changed_path = {
        "a.txt": {"sha256": "1", "bytes": 1},
        "c.txt": {"sha256": "2", "bytes": 2},
    }
    assert inventory_identity(first) == inventory_identity(second)
    assert inventory_identity(first) != inventory_identity(changed_path)


def test_inventory_identity_ignores_compact_packaging_policy():
    included = {
        "a.bin": {
            "sha256": "1",
            "bytes": 1,
            "category": "source",
            "included_in_compact_bundle": True,
        }
    }
    indexed = {
        "a.bin": {
            "sha256": "1",
            "bytes": 1,
            "category": "data",
            "included_in_compact_bundle": False,
        }
    }
    assert inventory_identity(included) == inventory_identity(indexed)


def test_deterministic_zip_is_verified_and_rejects_extra_member(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("alpha\n", encoding="utf-8")
    (source / "b.txt").write_text("beta\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_deterministic_zip(source, first)
    write_deterministic_zip(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert verify_zip_archive(first, source)["members"] == 2

    with zipfile.ZipFile(first, "a") as archive:
        archive.writestr("unexpected.txt", "unexpected")
    with pytest.raises(ValueError, match="membership"):
        verify_zip_archive(first, source)


def test_build_has_content_addressed_id_companion_check_and_reproducible_zip(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "README.md").write_text("project readme\n", encoding="utf-8")
    (project / "table.json").write_text('{"result":1}\n', encoding="utf-8")
    (project / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "src" / "prediction.npz").write_bytes(
        b"x" * (artifact_builder.SMALL_BINARY_LIMIT + 1)
    )
    monkeypatch.setattr(artifact_builder, "ROOT", project)
    monkeypatch.setattr(artifact_builder, "SOURCE_ROOTS", ("src",))
    monkeypatch.setattr(artifact_builder, "SOURCE_FILES", ("README.md", "table.json"))
    monkeypatch.setattr(artifact_builder, "CONTRACT_ROOTS", ())
    monkeypatch.setattr(artifact_builder, "EVIDENCE_ROOTS", ())
    monkeypatch.setattr(artifact_builder, "DATA_ROOTS", ())
    monkeypatch.setattr(artifact_builder, "TABLE_SOURCES", {"table": "table.json"})
    monkeypatch.setattr(artifact_builder, "run_text", lambda _command: "stable-version")
    monkeypatch.setattr(artifact_builder, "git_head", lambda: "a" * 40)

    first_output = tmp_path / "build-one" / "artifact"
    second_output = tmp_path / "build-two" / "artifact"
    first_output.parent.mkdir()
    second_output.parent.mkdir()
    first_zip = tmp_path / "build-one" / "artifact.zip"
    second_zip = tmp_path / "build-two" / "artifact.zip"
    first_manifest = artifact_builder.build(first_output, first_zip)
    second_manifest = artifact_builder.build(second_output, second_zip)

    assert first_manifest == second_manifest
    assert "created_unix" not in first_manifest
    assert first_manifest["artifact_id"] == (
        "airproof-reviewer-revision-sha256-" + first_manifest["source_tree_sha256"]
    )
    assert first_zip.read_bytes() == second_zip.read_bytes()
    assert artifact_builder.zip_checksum_path(first_zip).read_text(encoding="ascii").split()[0] == sha256_file(first_zip)
    assert (first_output / "README.md").read_text(encoding="utf-8") == "project readme\n"
    assert (first_output / "REVIEWER_ARTIFACT_README.md").is_file()

    compact = subprocess.run(
        [sys.executable, str(first_output / "verify_artifact.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    compact_result = json.loads(compact.stdout)
    assert compact.returncode == 0
    assert compact_result["compact_status"] == "pass"
    assert compact_result["release_complete"] is False
    complete = subprocess.run(
        [
            sys.executable,
            str(first_output / "verify_artifact.py"),
            "--companion-root",
            str(project),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    complete_result = json.loads(complete.stdout)
    assert complete.returncode == 0
    assert complete_result["external_objects_status"] == "pass"
    assert complete_result["release_complete"] is True
