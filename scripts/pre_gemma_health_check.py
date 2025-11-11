#!/usr/bin/env python3
"""Pre-GEMMA readiness checks for repository, workflows, deps, and artifacts."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "diagnostics" / "gemma_readiness_report.json"


class CheckResult(TypedDict):
    name: str
    status: str
    details: str


class ReadinessReport(TypedDict):
    generated_at: str
    checks: list[CheckResult]
    warnings: list[str]
    errors: list[str]
    migration_plan: list[str]

report: ReadinessReport = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "checks": [],
    "warnings": [],
    "errors": [],
    "migration_plan": [],
}


def add_migration_item(message: str) -> None:
    if message not in report["migration_plan"]:
        report["migration_plan"].append(message)


def record_check(name: str, status: str, details: str) -> None:
    entry: CheckResult = {"name": name, "status": status, "details": details}
    report["checks"].append(entry)
    if status == "fail":
        report["errors"].append(details)
    elif status == "warn":
        report["warnings"].append(details)


def check_repo_structure() -> None:
    expected = [
        "src/ml",
        "src/ml/adapters/gemma",
        "src/ml/features",
        "src/ml/integration",
        "data/models/gemma",
        "data/cache/gemma/scalers",
        "features/gemma/selected",
        "diagnostics/gemma",
        "logs/gemma",
    ]
    missing = [path for path in expected if not (REPO_ROOT / path).exists()]
    if missing:
        record_check(
            "repository_structure",
            "fail",
            f"Missing directories: {', '.join(missing)}.",
        )
        add_migration_item("Bootstrap GEMMA directory tree via setup script (or create manually).")
    else:
        record_check("repository_structure", "pass", "All GEMMA scaffolding directories are present.")


def collect_python_versions(text: str) -> list[str]:
    versions: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "python-version" not in stripped:
            continue
        if stripped.startswith("echo "):
            continue
        key, _, remainder = stripped.partition(":")
        if key.strip() != "python-version":
            continue
        value = remainder.strip()
        if not value:
            continue
        versions.append(value.strip('\"\''))
    return versions


def check_workflow_python_version() -> None:
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    non_compliant: dict[str, list[str]] = {}
    dynamic: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml*")):
        text = path.read_text(encoding="utf-8")
        versions = collect_python_versions(text)
        if not versions:
            continue
        static_versions = [value for value in versions if not value.startswith("${{")]
        dynamic_versions = [value for value in versions if value.startswith("${{")]
        if static_versions and all(version.startswith("3.11") for version in static_versions):
            if dynamic_versions:
                dynamic.append(path.name)
            continue
        if dynamic_versions and not static_versions:
            dynamic.append(path.name)
        else:
            non_compliant[path.name] = versions
    if non_compliant:
        issues = "; ".join(f"{name}: {values}" for name, values in non_compliant.items())
        record_check(
            "workflow_python",
            "fail",
            f"Non-3.11 python-version detected -> {issues}",
        )
        add_migration_item("Pin python-version: 3.11 on all GitHub workflows.")
    elif dynamic:
        record_check(
            "workflow_python",
            "warn",
            f"Dynamic python-version detected in {', '.join(dynamic)}; ensure it resolves to 3.11.",
        )
        add_migration_item("Lock dynamic workflow python versions to 3.11 where possible.")
    else:
        record_check("workflow_python", "pass", "All workflows pin python-version to 3.11.")


def load_dependency_names() -> set[str]:
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    names: set[str] = set()
    for dep in deps:
        cleaned = dep.split(";")[0].strip()
        token = re.split(r"[<>=!~\[]", cleaned, maxsplit=1)[0].strip().lower()
        if token:
            names.add(token)
    return names


def check_ml_dependencies() -> None:
    required = {"scikit-learn", "torch", "xgboost"}
    present = load_dependency_names()
    missing = sorted(dep for dep in required if dep not in present)
    if missing:
        detail = f"Missing ML dependencies in pyproject.toml: {', '.join(missing)}."
        record_check("ml_dependencies", "warn", detail)
        add_migration_item(f"Add GEMMA training deps ({', '.join(missing)}) to pyproject.toml.")
    else:
        record_check("ml_dependencies", "pass", "Key ML dependencies already declared.")


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def check_model_and_scaler_artifacts() -> None:
    model_root = REPO_ROOT / "data" / "models" / "gemma"
    scaler_root = REPO_ROOT / "data" / "cache" / "gemma" / "scalers"
    model_files = count_files(model_root)
    scaler_files = count_files(scaler_root)

    if model_files == 0:
        record_check("gemma_models", "warn", "No GEMMA model artifacts detected in data/models/gemma.")
        add_migration_item("Train or import baseline GEMMA models into data/models/gemma.")
    else:
        record_check("gemma_models", "pass", f"Detected {model_files} model artifact(s).")

    if scaler_files == 0:
        record_check("gemma_scalers", "warn", "Scaler snapshots missing under data/cache/gemma/scalers.")
        add_migration_item("Generate feature scalers and store them under data/cache/gemma/scalers.")
    else:
        record_check("gemma_scalers", "pass", f"Detected {scaler_files} scaler artifact(s).")


def main() -> int:
    check_repo_structure()
    check_workflow_python_version()
    check_ml_dependencies()
    check_model_and_scaler_artifacts()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"GEMMA readiness report stored at {REPORT_PATH.relative_to(REPO_ROOT)}")
    if report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
