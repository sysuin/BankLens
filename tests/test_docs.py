"""Phase 7: the governance story must be real, not a paragraph.

Governance documents exist and say the things the doctrines require; the
README's coverage table points at files that exist; the bias check passes
with no model and no key; the deployment stubs parse.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "phrases"),
    [
        (
            "docs/governance/RESPONSIBLE_AI.md",
            ["decide", "narrat", "kill switch", "bias", "limitation"],
        ),
        ("docs/governance/MODEL_CARD.md", ["intended use", "limitation", "eval"]),
        ("docs/governance/DATA_RETENTION.md", ["retention", "deletion", "synthetic"]),
        ("CLAUDE.md", ["deterministic", "make gate", "do not"]),
    ],
)
def test_governance_documents_exist_and_cover_the_required_points(
    path: str, phrases: list[str]
) -> None:
    text = _read(path).lower()
    missing = [p for p in phrases if p not in text]
    assert not missing, f"{path} is missing {missing}"


def test_readme_has_the_interview_script_and_limitations() -> None:
    readme = _read("README.md")
    assert "## How to demo this in an interview" in readme
    assert "## Limitations, honestly" in readme
    assert "## Where each concept lives" in readme


def test_readme_coverage_table_points_at_real_files() -> None:
    readme = _read("README.md")
    start = readme.index("## Where each concept lives")
    end = readme.index("\n## ", start + 10)
    table = readme[start:end]
    paths = set()
    for row in table.splitlines():
        if not row.startswith("| ") or row.startswith("| Topic") or "---" in row:
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        paths.update(re.findall(r"`([^`]+)`", cells[1]))
    assert len(paths) >= 30
    missing = [p for p in sorted(paths) if not (ROOT / p).exists()]
    assert not missing, f"coverage table names files that do not exist: {missing}"


def test_bias_check_passes_without_a_model() -> None:
    from evals.bias_check import main

    assert main() == 0


def test_kubernetes_manifest_parses_and_names_the_three_tiers() -> None:
    docs = list(yaml.safe_load_all(_read("deploy/k8s/banklens.yaml")))
    kinds = {(d["kind"], d["metadata"]["name"]) for d in docs if d}
    for tier in ("api", "worker", "console"):
        assert ("Deployment", f"banklens-{tier}") in kinds
    assert ("Namespace", "banklens") in kinds


def test_terraform_stub_describes_the_host_and_registry() -> None:
    tf = _read("deploy/terraform/main.tf")
    for resource in (
        'resource "aws_ecr_repository"',
        'resource "aws_security_group"',
        'resource "aws_instance"',
    ):
        assert resource in tf
    assert "volume_size = 30" in tf


def test_api_version_is_the_platform_release() -> None:
    from app.api.routes.health import VERSION

    assert VERSION.startswith("1.")
