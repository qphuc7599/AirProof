import json
import subprocess
import sys

import scripts.build_standalone_release_artifact as artifact_builder


def test_gate_b_development_and_result_bearing_table_sources_are_selected():
    assert "reports/v8/m5_release_assimilation_development_v1" in (
        artifact_builder.EVIDENCE_ROOTS
    )
    assert artifact_builder.TABLE_SOURCES["v4_fairness_frontier"].endswith(
        "/selection.json"
    )
    assert artifact_builder.TABLE_SOURCES["v4_privacy_grid"].endswith(
        "/analysis.json"
    )
    assert artifact_builder.TABLE_SOURCES["v4_beijing_seasonal"].endswith(
        "/analysis.json"
    )
    assert artifact_builder.TABLE_SOURCES["v4_dtn_confirmation"].endswith(
        "/summary.json"
    )


def test_gate_c_completion_binds_lock_and_outputs():
    completion_path = (
        artifact_builder.ROOT
        / "reports/v8/external_wrapper_gate_c_v1/completion.json"
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    root = completion_path.parent
    lock = root / completion["confirmation_lock"]
    assert artifact_builder.sha256_file(lock) == completion["confirmation_lock_sha256"]
    for relative, record in completion["outputs"].items():
        path = root / relative
        assert path.stat().st_size == record["bytes"]
        assert artifact_builder.sha256_file(path) == record["sha256"]


def test_build_materializes_and_independently_verifies_companion(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "README.md").write_text("project\n", encoding="utf-8")
    (project / "table.json").write_text('{"value": 1}\n', encoding="utf-8")
    (project / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "src" / "prediction.npz").write_bytes(b"large-object")

    monkeypatch.setattr(artifact_builder, "ROOT", project)
    monkeypatch.setattr(artifact_builder, "SOURCE_ROOTS", ("src",))
    monkeypatch.setattr(
        artifact_builder, "SOURCE_FILES", ("README.md", "table.json")
    )
    monkeypatch.setattr(artifact_builder, "CONTRACT_ROOTS", ())
    monkeypatch.setattr(artifact_builder, "EVIDENCE_ROOTS", ())
    monkeypatch.setattr(artifact_builder, "DATA_ROOTS", ())
    monkeypatch.setattr(artifact_builder, "TABLE_SOURCES", {"table": "table.json"})
    monkeypatch.setattr(artifact_builder, "SMALL_BINARY_LIMIT", 8)
    monkeypatch.setattr(artifact_builder, "git_head", lambda: None)

    def fake_run_text(command):
        if command[-2:] == ["pip", "freeze"]:
            return f"demo==1.0\n-e {project}"
        return "test-tool 1.0"

    monkeypatch.setattr(artifact_builder, "run_text", fake_run_text)

    compact = tmp_path / "compact"
    compact_zip = tmp_path / "compact.zip"
    companion = tmp_path / "companion"
    companion_zip = tmp_path / "companion.zip"
    manifest = artifact_builder.build(
        compact,
        compact_zip,
        companion,
        companion_zip,
    )

    freeze = (compact / "environment/pip-freeze.txt").read_text(encoding="utf-8")
    assert "-e ." in freeze
    assert str(project) not in freeze
    assert (companion / "src/prediction.npz").read_bytes() == b"large-object"
    assert manifest["companion_object_index"]["archive"]["sha256"] == (
        artifact_builder.sha256_file(companion_zip)
    )
    assert (tmp_path / "compact.zip.sha256").is_file()
    assert (tmp_path / "companion.zip.sha256").is_file()

    companion_check = subprocess.run(
        [sys.executable, str(companion / "verify_companion.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert companion_check.returncode == 0, companion_check.stdout + companion_check.stderr
    assert json.loads(companion_check.stdout)["status"] == "pass"

    incomplete = subprocess.run(
        [sys.executable, str(compact / "verify_artifact.py"), "--require-complete"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert incomplete.returncode == 1
    incomplete_result = json.loads(incomplete.stdout)
    assert incomplete_result["compact_status"] == "pass"
    assert incomplete_result["release_complete"] is False

    complete = subprocess.run(
        [
            sys.executable,
            str(compact / "verify_artifact.py"),
            "--companion-root",
            str(companion),
            "--require-complete",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert complete.returncode == 0, complete.stdout + complete.stderr
    complete_result = json.loads(complete.stdout)
    assert complete_result["external_objects_status"] == "pass"
    assert complete_result["release_complete"] is True
