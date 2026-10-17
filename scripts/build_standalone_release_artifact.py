"""Build the compact, executable artifact for the standalone AirProof release.

The compact archive contains code, configuration, tests, contracts, manuscript
sources, metric rows and campaign locks.  Large binary predictions and downloaded
archives remain content-addressed in the workspace and are enumerated in the
manifest so that a repository/Zenodo release can upload them as a second object
without changing any scientific identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOTS = (
    "airproof",
    "scripts",
    "tests",
    "configs",
    "docs",
    "overleaf",
)

SOURCE_FILES = (
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "contracts/package.json",
    "contracts/package-lock.json",
    "contracts/hardhat.config.js",
    "contracts/README.md",
)

CONTRACT_ROOTS = (
    "contracts/src",
    "contracts/test",
    "contracts/scripts",
)

EVIDENCE_ROOTS = (
    "reports/v4_primary/core_final_7600_7629_20260903",
    "reports/v4_validation/core_selected_7300_7311",
    "reports/v4_validation/fairness_frontier_7400_7407",
    "reports/v4_validation/privacy_full_grid_7500_7529",
    "reports/v4_validation/beijing_seasonal_20260903",
    "reports/v4_validation/esntnn_purpleair_7800",
    "reports/v4_validation/esntnn_wrapper_7850_7861",
    "reports/v4_validation/dtn_deadline_7900_7907",
    "reports/v4_validation/dtn_deadline_confirm_7950_7979",
    "reports/v4_validation/empirical_contacts_8050_8054",
    "reports/v4_validation/ingestion_concurrency_20260903",
    "reports/v4_validation/primary_chain_20260903",
    "reports/v4_validation/validation_chain_20260903",
    "reports/v4_validation/sepolia_30_anchors_20260903",
    "reports/v5_diagnostics/selected_archive_factorial_916000_916001",
    "reports/v5_diagnostics/wrapper_916000_916001",
    "reports/v5_diagnostics/original_wrapper_tail_7850_7861.json",
    "reports/v5/stress_registered_v1/scoring",
    "reports/v5/stress_export",
    "reports/v6/lifetime_budget_repair_development_v5",
    "reports/v6/lifetime_budget7_validation_v2",
    "reports/v6/fairness_validation_v1",
    "reports/v6/release_incremental_value",
    "reports/v6/privacy_release_contract",
    "reports/v6/uncertainty_development_validation_v2",
    "reports/v6/uncertainty_event_v3_1",
    "reports/v6/accountability_integrated_confirmation",
    "reports/v6/integration_smoke_accountability_batch_v3",
    "reports/v6/durable_inclusion_async_return_v1",
    "reports/v6/durable_public_checkpoint_v2",
    "reports/v6/ledger_transparency_same_stream_v2",
    "reports/v7/reviewer_revision",
    "reports/v8/m5_release_assimilation_development_v1",
    "reports/v8/m5_release_reference_stress_confirmation_v1",
    "reports/v8/external_wrapper_gate_c_v1",
    "reports/v8/development/public_checkpoint_smoke_v1",
    "reports/v8/gate_a_obstruction_audit_20260914.md",
)

DATA_ROOTS = (
    "data",
)

ALWAYS_COPY_SUFFIXES = {
    ".bib", ".cfg", ".csv", ".json", ".jsonl", ".lock", ".md", ".py",
    ".ps1", ".sol", ".tex", ".toml", ".txt", ".yaml", ".yml", ".js",
}
SMALL_BINARY_SUFFIXES = {".npz", ".npy", ".pdf", ".png", ".rds", ".pkl"}
SMALL_BINARY_LIMIT = 2 * 1024 * 1024
ARTIFACT_ID_PREFIX = "airproof-standalone-release"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

TABLE_SOURCES = {
    "v4_primary_H1_H7": "reports/v4_primary/core_final_7600_7629_20260903/analysis.json",
    "v4_fairness_frontier": "reports/v4_validation/fairness_frontier_7400_7407/selection.json",
    "v4_privacy_grid": "reports/v4_validation/privacy_full_grid_7500_7529/analysis.json",
    "v4_beijing_seasonal": "reports/v4_validation/beijing_seasonal_20260903/analysis.json",
    "v4_esntnn_wrapper": "reports/v4_validation/esntnn_wrapper_7850_7861/summary.json",
    "v5_wrapper_factorial": "reports/v5_diagnostics/wrapper_916000_916001/summary.json",
    "v5_wrapper_failure_partition": "reports/v5_diagnostics/original_wrapper_tail_7850_7861.json",
    "v5_reference_attack_stress": "reports/v5/stress_export/means.csv",
    "v4_dtn_confirmation": "reports/v4_validation/dtn_deadline_confirm_7950_7979/summary.json",
    "v4_sepolia": "reports/v4_validation/sepolia_30_anchors_20260903/anchors.json",
    "v6_lifetime_budget7": "reports/v6/lifetime_budget7_validation_v2/outcomes/analysis.json",
    "v6_fairness_extension": "reports/v6/fairness_validation_v1/analysis.json",
    "v6_release_incremental_value": "reports/v6/release_incremental_value/result.json",
    "v6_uncertainty_event": "reports/v6/uncertainty_event_v3_1/results.json",
    "v6_accountability": "reports/v6/accountability_integrated_confirmation/results.json",
    "v6_transparency": "reports/v6/ledger_transparency_same_stream_v2/analysis.json",
    "v7_lifetime_certificate": "reports/v7/reviewer_revision/lifetime_retained_diagnostic/certificate.json",
    "v7_lifetime_phase_audit": "reports/v7/reviewer_revision/lifetime_retained_diagnostic/paired_phase_summary.json",
    "v7_lifetime_equal_exposure": "reports/v7/reviewer_revision/lifetime_uniform_exposure_control_v1/analysis.json",
    "v7_shared_lifetime_diagnostic": "reports/v7/reviewer_revision/shared_resource_confirmation_v2/analysis_detail.json",
    "v7_shared_core_confirmation": "reports/v7/reviewer_revision/shared_resource_core_confirmation_v3/analysis_detail.json",
    "v7_shared_core_fairness": "reports/v7/reviewer_revision/shared_core_fairness_control_v1/analysis.json",
    "v7_shared_core_component_controls": "reports/v7/reviewer_revision/shared_core_component_controls_v1/analysis.json",
    "v7_event_intervals": "reports/v7/reviewer_revision/event_interval_confirmation_v1/analysis.json",
    "v7_modern_relay": "reports/v7/reviewer_revision/retained_modern_relay_comparison/analysis.json",
    "release_reference_stress_confirmation": "reports/v8/m5_release_reference_stress_confirmation_v1/analysis.json",
    "external_equal_information_confirmation": "reports/v8/external_wrapper_gate_c_v1/confirmation_v1/summary.json",
    "public_checkpoint_engineering_smoke": "reports/v8/development/public_checkpoint_smoke_v1/smoke_summary.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_identity(inventory: dict[str, dict]) -> str:
    """Return the identity of paths and bytes, independent of packaging policy."""
    content_inventory = {
        name: {"bytes": record["bytes"], "sha256": record["sha256"]}
        for name, record in inventory.items()
    }
    canonical = json.dumps(content_inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def zip_checksum_path(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.name + ".sha256")


def write_deterministic_zip(source_root: Path, zip_path: Path) -> None:
    """Write a byte-reproducible ZIP from source_root using normalized metadata."""
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
        for path in sorted(candidate for candidate in source_root.rglob("*") if candidate.is_file()):
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_zip_archive(zip_path: Path, source_root: Path) -> dict:
    """Verify member safety, membership, CRC and bytes against the staged directory."""
    expected = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in source_root.rglob("*") if path.is_file()
    }
    failures: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            failures.append("duplicate ZIP member")
        unsafe = [
            name for name in names
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or "\\" in name
        ]
        if unsafe:
            failures.append(f"unsafe ZIP members: {unsafe}")
        if set(names) != set(expected):
            failures.append("ZIP membership differs from staged artifact")
        bad_crc = archive.testzip()
        if bad_crc is not None:
            failures.append(f"ZIP CRC failure: {bad_crc}")
        for name in sorted(set(names) & set(expected)):
            digest = hashlib.sha256(archive.read(name)).hexdigest()
            if digest != expected[name]:
                failures.append(f"ZIP content mismatch: {name}")
    if failures:
        raise ValueError("; ".join(failures))
    return {"members": len(expected), "sha256": sha256_file(zip_path)}


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, cwd=ROOT, check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    text = (result.stdout or result.stderr).strip()
    return text if text else f"exit={result.returncode}"


def normalize_pip_freeze(text: str) -> str:
    """Replace the workspace's absolute editable install with a portable path."""
    normalized: list[str] = []
    root = ROOT.resolve()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-e "):
            target = stripped[3:].strip()
            try:
                is_workspace = Path(target).resolve() == root
            except (OSError, ValueError):
                is_workspace = False
            if is_workspace:
                prefix = line[: len(line) - len(line.lstrip())]
                line = f"{prefix}-e ."
        normalized.append(line)
    return "\n".join(normalized)


def iter_files(relative: str):
    path = ROOT / relative
    if path.is_file():
        yield path
    elif path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and "__pycache__" not in candidate.parts:
                yield candidate


def include_in_compact(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in ALWAYS_COPY_SUFFIXES:
        return True
    return bool(suffix in SMALL_BINARY_SUFFIXES and path.stat().st_size <= SMALL_BINARY_LIMIT)


def copy_one(source: Path, destination_root: Path) -> None:
    relative = source.relative_to(ROOT)
    target = destination_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def verify_object_tree(root: Path, objects: dict[str, dict]) -> None:
    failures: list[str] = []
    for name, record in objects.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            failures.append(f"unsafe companion path: {name}")
            continue
        path = root / relative
        if not path.is_file():
            failures.append(f"missing companion object: {name}")
        elif path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            failures.append(f"companion object mismatch: {name}")
    if failures:
        raise ValueError("; ".join(failures))


COMPANION_VERIFIER = r'''#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

index = json.loads((ROOT / "COMPANION_OBJECTS.json").read_text(encoding="utf-8"))
failures = []
checked = 0
for name, record in index["objects"].items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        failures.append(f"unsafe companion path: {name}")
        continue
    path = ROOT / relative
    if not path.is_file():
        failures.append(f"missing: {name}")
    elif path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        failures.append(f"mismatch: {name}")
    else:
        checked += 1
result = {
    "status": "pass" if not failures else "fail",
    "source_tree_sha256": index["source_tree_sha256"],
    "objects_checked": checked,
    "expected_objects": len(index["objects"]),
    "failures": failures,
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if not failures else 1)
'''


def build(
    output: Path,
    zip_path: Path | None,
    companion_output: Path | None = None,
    companion_zip_path: Path | None = None,
) -> dict:
    output = output.resolve()
    zip_path = zip_path.resolve() if zip_path is not None else None
    companion_output = (
        companion_output.resolve() if companion_output is not None else None
    )
    companion_zip_path = (
        companion_zip_path.resolve() if companion_zip_path is not None else None
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    if companion_zip_path is not None and companion_output is None:
        raise ValueError("--companion-zip requires --companion-output")
    if companion_output is not None and companion_output == output:
        raise ValueError("compact and companion output directories must differ")
    if companion_output is not None and (
        companion_output.is_relative_to(output) or output.is_relative_to(companion_output)
    ):
        raise ValueError("compact and companion output directories must not be nested")
    if zip_path is not None and companion_zip_path is not None and zip_path == companion_zip_path:
        raise ValueError("compact and companion ZIP paths must differ")
    if zip_path is not None and zip_path.is_relative_to(output):
        raise ValueError("compact ZIP must be outside the compact staging directory")
    if (
        companion_output is not None
        and companion_zip_path is not None
        and companion_zip_path.is_relative_to(companion_output)
    ):
        raise ValueError("companion ZIP must be outside the companion staging directory")
    if zip_path is not None and zip_path.exists():
        raise FileExistsError(f"refusing to overwrite zip: {zip_path}")
    if zip_path is not None and zip_checksum_path(zip_path).exists():
        raise FileExistsError(
            f"refusing to overwrite zip checksum: {zip_checksum_path(zip_path)}"
        )
    if companion_output is not None and companion_output.exists():
        raise FileExistsError(f"refusing to overwrite companion directory: {companion_output}")
    if companion_zip_path is not None and companion_zip_path.exists():
        raise FileExistsError(f"refusing to overwrite companion zip: {companion_zip_path}")
    if companion_zip_path is not None and zip_checksum_path(companion_zip_path).exists():
        raise FileExistsError(
            f"refusing to overwrite companion zip checksum: {zip_checksum_path(companion_zip_path)}"
        )
    output.mkdir(parents=True)
    inventory: dict[str, dict] = {}
    categories = {
        "source": list(SOURCE_FILES) + list(SOURCE_ROOTS) + list(CONTRACT_ROOTS),
        "evidence": list(EVIDENCE_ROOTS),
        "data": list(DATA_ROOTS),
    }
    seen: set[Path] = set()
    for category, roots in categories.items():
        for relative in roots:
            for path in iter_files(relative):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                rel = path.relative_to(ROOT).as_posix()
                copied = include_in_compact(path)
                inventory[rel] = {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "category": category,
                    "included_in_compact_bundle": copied,
                }
                if copied:
                    copy_one(path, output)

    environment = output / "environment"
    environment.mkdir()
    (environment / "python.txt").write_text(
        json.dumps({
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        }, indent=2), encoding="utf-8",
    )
    (environment / "pip-freeze.txt").write_text(
        normalize_pip_freeze(run_text([sys.executable, "-m", "pip", "freeze"])) + "\n",
        encoding="utf-8",
    )
    (environment / "latex-version.txt").write_text(
        run_text(["pdflatex", "--version"]) + "\n", encoding="utf-8"
    )
    (environment / "node-version.txt").write_text(
        run_text(["node", "--version"]) + "\n", encoding="utf-8"
    )

    missing_table_sources = [
        relative for relative in TABLE_SOURCES.values() if not (ROOT / relative).is_file()
    ]
    if missing_table_sources:
        raise FileNotFoundError(f"missing manuscript table sources: {missing_table_sources}")
    source_tree_sha256 = inventory_identity(inventory)
    external_objects = {
        name: {key: record[key] for key in ("bytes", "category", "sha256")}
        for name, record in inventory.items()
        if not record["included_in_compact_bundle"]
    }
    companion_index = {
        "schema_version": 1,
        "source_tree_sha256": source_tree_sha256,
        "objects": external_objects,
    }
    companion_path = output / "COMPANION_OBJECTS.json"
    companion_path.write_text(
        json.dumps(companion_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    companion_archive = None
    companion_directory = None
    if companion_output is not None:
        companion_output.mkdir(parents=True)
        for name in external_objects:
            copy_one(ROOT / name, companion_output)
        (companion_output / "COMPANION_OBJECTS.json").write_text(
            json.dumps(companion_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (companion_output / "verify_companion.py").write_text(
            COMPANION_VERIFIER, encoding="utf-8"
        )
        verify_object_tree(companion_output, external_objects)
        companion_directory = {
            "name": companion_output.name,
            "objects_verified": len(external_objects),
        }
        if companion_zip_path is not None:
            companion_zip_path.parent.mkdir(parents=True, exist_ok=True)
            write_deterministic_zip(companion_output, companion_zip_path)
            companion_zip_result = verify_zip_archive(companion_zip_path, companion_output)
            companion_checksum_path = zip_checksum_path(companion_zip_path)
            companion_checksum_path.write_text(
                f"{companion_zip_result['sha256']}  {companion_zip_path.name}\n",
                encoding="ascii",
            )
            companion_archive = {
                "name": companion_zip_path.name,
                "bytes": companion_zip_path.stat().st_size,
                "sha256": companion_zip_result["sha256"],
                "checksum_name": companion_checksum_path.name,
                "members": companion_zip_result["members"],
            }
    manifest = {
        "schema_version": 2,
        "artifact_id": f"{ARTIFACT_ID_PREFIX}-sha256-{source_tree_sha256}",
        "source_tree_sha256": source_tree_sha256,
        "git_head": git_head(),
        "scope": "compact executable artifact plus content-addressed large-object index",
        "manuscript_source": "overleaf/paper.tex",
        "entry_points": {
            "python_tests": "python -m pytest -q",
            "python_smoke": "python -m airproof.cli fixtures",
            "solidity_tests": "cd contracts && npm ci && npm test",
            "manuscript": "cd overleaf && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex",
        },
        "testnet": {
            "network": "ethereum-sepolia-public-testnet",
            "contract_address": "0x7ae1AEC092a8E399F529db8988C7fC7479e052D0",
            "transaction_file": "reports/v4_validation/sepolia_30_anchors_20260903/anchors.json",
            "transaction_count": 30,
        },
        "table_sources": TABLE_SOURCES,
        "files": inventory,
        "included_files": sum(v["included_in_compact_bundle"] for v in inventory.values()),
        "indexed_large_or_binary_files": sum(
            not v["included_in_compact_bundle"] for v in inventory.values()
        ),
        "companion_object_index": {
            "path": "COMPANION_OBJECTS.json",
            "sha256": sha256_file(companion_path),
            "objects": len(external_objects),
            "required_for_full_reproduction": bool(external_objects),
            "materialized_directory": companion_directory,
            "archive": companion_archive,
            "publication_status": (
                "materialized_locally_not_published"
                if companion_output is not None
                else "not_materialized_or_published_by_this_builder"
            ),
        },
        "limitations": (
            "The compact archive indexes but does not duplicate large binary predictions and "
            "downloaded archives.  A public repository release must upload every indexed object "
            "or publish a second large-object archive with the same hashes."
        ),
    }
    raw = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(raw + "\n", encoding="utf-8")
    (output / "ARTIFACT_SHA256.txt").write_text(
        sha256_file(manifest_path) + "  artifact_manifest.json\n",
        encoding="ascii",
    )
    (output / "SOURCE_TREE_SHA256.txt").write_text(
        manifest["source_tree_sha256"] + "  content-addressed source/evidence inventory\n",
        encoding="ascii",
    )
    verifier = r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"unsafe manifest path: {name}")
    return root / relative

parser = argparse.ArgumentParser()
parser.add_argument(
    "--companion-root",
    help="root containing every object listed in COMPANION_OBJECTS.json",
)
parser.add_argument(
    "--require-complete",
    action="store_true",
    help="fail unless every indexed companion object is supplied and verified",
)
args = parser.parse_args()
manifest_path = ROOT / "artifact_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_manifest = (ROOT / "ARTIFACT_SHA256.txt").read_text(encoding="ascii").split()[0]
failures = []
if digest(manifest_path) != expected_manifest:
    failures.append("artifact_manifest.json checksum mismatch")
content_inventory = {
    name: {"bytes": record["bytes"], "sha256": record["sha256"]}
    for name, record in manifest["files"].items()
}
canonical = json.dumps(content_inventory, sort_keys=True, separators=(",", ":"))
source_tree_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
if source_tree_sha256 != manifest["source_tree_sha256"]:
    failures.append("source-tree identity mismatch")
if manifest["artifact_id"] != f"airproof-standalone-release-sha256-{source_tree_sha256}":
    failures.append("artifact_id is not bound to source-tree identity")
for name, record in manifest["files"].items():
    if not record["included_in_compact_bundle"]:
        continue
    try:
        path = safe_path(ROOT, name)
    except ValueError as error:
        failures.append(str(error))
        continue
    if not path.is_file():
        failures.append(f"missing: {name}")
    elif path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        failures.append(f"mismatch: {name}")
index_record = manifest["companion_object_index"]
index_path = ROOT / index_record["path"]
companion_index = json.loads(index_path.read_text(encoding="utf-8"))
if digest(index_path) != index_record["sha256"]:
    failures.append("COMPANION_OBJECTS.json checksum mismatch")
external_records = {
    name: {key: record[key] for key in ("bytes", "category", "sha256")}
    for name, record in manifest["files"].items()
    if not record["included_in_compact_bundle"]
}
if companion_index.get("source_tree_sha256") != source_tree_sha256:
    failures.append("companion index source-tree identity mismatch")
if companion_index.get("objects") != external_records:
    failures.append("companion index differs from manifest")
external_checked = 0
compact_failures = len(failures)
if args.companion_root:
    companion_root = Path(args.companion_root).resolve()
    for name, record in external_records.items():
        try:
            path = safe_path(companion_root, name)
        except ValueError as error:
            failures.append(str(error))
            continue
        if not path.is_file():
            failures.append(f"missing companion object: {name}")
        elif path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            failures.append(f"companion object mismatch: {name}")
        else:
            external_checked += 1
if args.require_complete and external_checked != len(external_records):
    failures.append(
        "complete release required but not every indexed companion object was verified"
    )
result = {
    "status": "pass" if not failures else "fail",
    "compact_status": "pass" if compact_failures == 0 else "fail",
    "artifact_id": manifest["artifact_id"],
    "source_tree_sha256": manifest["source_tree_sha256"],
    "included_files_checked": sum(
        record["included_in_compact_bundle"] for record in manifest["files"].values()
    ),
    "indexed_external_objects": sum(
        not record["included_in_compact_bundle"] for record in manifest["files"].values()
    ),
    "external_objects_checked": external_checked,
    "release_complete": not failures and external_checked == len(external_records),
    "external_objects_status": (
        "pass" if external_checked == len(external_records)
        else "not_checked" if not args.companion_root
        else "fail"
    ),
    "failures": failures,
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if not failures else 1)
'''
    (output / "verify_artifact.py").write_text(verifier, encoding="utf-8")
    readme = """# AirProof research artifact

This compact artifact contains the executable source, configurations, tests,
contract sources, manuscript, experiment configurations and compact result rows.
`artifact_manifest.json` records the SHA-256 and byte length of every included
or indexed object. Large prediction arrays and downloaded archives are indexed
rather than duplicated and can be distributed as a companion large-object archive.

Run `python verify_artifact.py` for the compact bundle. Run
`python verify_artifact.py --companion-root <directory> --require-complete`
before claiming a complete release; the command checks every indexed external
object and fails if the companion is absent or incomplete. A materialized
companion also contains `verify_companion.py` for an independent object check.
Then run
`python -m pytest -q` and compile `overleaf/paper.tex`. The Sepolia
contract address and all 30 transaction hashes are in the manifest and named
evidence file.  No command in this artifact sends a new public transaction.
"""
    (output / "ARTIFACT_README.md").write_text(readme, encoding="utf-8")

    if zip_path is not None:
        if zip_path.exists():
            raise FileExistsError(f"refusing to overwrite zip: {zip_path}")
        checksum_path = zip_checksum_path(zip_path)
        if checksum_path.exists():
            raise FileExistsError(f"refusing to overwrite zip checksum: {checksum_path}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        write_deterministic_zip(output, zip_path)
        zip_result = verify_zip_archive(zip_path, output)
        checksum_path.write_text(
            f"{zip_result['sha256']}  {zip_path.name}\n", encoding="ascii"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip")
    parser.add_argument(
        "--companion-output",
        help="materialize every indexed external object under this directory",
    )
    parser.add_argument(
        "--companion-zip",
        help="write and checksum a deterministic ZIP of --companion-output",
    )
    args = parser.parse_args()
    manifest = build(
        Path(args.output).resolve(),
        Path(args.zip).resolve() if args.zip else None,
        Path(args.companion_output).resolve() if args.companion_output else None,
        Path(args.companion_zip).resolve() if args.companion_zip else None,
    )
    result = {
        "artifact_id": manifest["artifact_id"],
        "included_files": manifest["included_files"],
        "indexed_large_or_binary_files": manifest["indexed_large_or_binary_files"],
        "output": str(Path(args.output).resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "companion_output": (
            str(Path(args.companion_output).resolve()) if args.companion_output else None
        ),
        "companion_zip": (
            str(Path(args.companion_zip).resolve()) if args.companion_zip else None
        ),
    }
    if args.zip:
        resolved_zip = Path(args.zip).resolve()
        result["zip_sha256"] = sha256_file(resolved_zip)
        result["zip_checksum"] = str(zip_checksum_path(resolved_zip))
    if args.companion_zip:
        resolved_companion_zip = Path(args.companion_zip).resolve()
        result["companion_zip_sha256"] = sha256_file(resolved_companion_zip)
        result["companion_zip_checksum"] = str(
            zip_checksum_path(resolved_companion_zip)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
