import json
from pathlib import Path
import sys

from tissue_growth.cli import main


def test_benchmark_dispatch_can_apply_config_overrides(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "tissue-growth",
        "benchmark",
        "--backend", "scipy",
        "--geometry", "sphere",
        "--sizes", "4",
        "--steps", "1",
        "--dt", "0.01",
    ])

    assert main() == 0
    assert '"measured_steps": 1' in capsys.readouterr().out


def test_aim1_dispatch_writes_machine_readable_report(monkeypatch, tmp_path, capsys):
    import tissue_growth.organization as organization

    expected = {
        "criteria": {"all_required_evidence_passed": True},
        "wall_seconds": 0.25,
    }
    monkeypatch.setattr(organization, "run_aim1_protocol", lambda protocol: expected)
    output = tmp_path/"aim1.json"
    config = Path(__file__).resolve().parents[1]/"configs"/"aim1_normal.json"
    monkeypatch.setattr(sys, "argv", [
        "tissue-growth", "aim1", "--config", str(config), "--output", str(output),
    ])

    assert main() == 0
    assert json.loads(output.read_text()) == expected
    assert '"all_required_evidence_passed": true' in capsys.readouterr().out
