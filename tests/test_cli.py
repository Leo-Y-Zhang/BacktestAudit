"""Tests for the lyra-validate command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lyravalidate.cli import main

_SAMPLE_CSV = Path(__file__).resolve().parents[1] / "examples" / "sample_returns.csv"


def _write_returns(path: Path, data: np.ndarray, columns: list[str]) -> None:
    pd.DataFrame(data, columns=columns).to_csv(path, index=False)


def test_cli_deployable_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(42)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    csv = tmp_path / "good.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])

    code = main([str(csv), "--n-trials", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "DEPLOYABLE" in out
    assert "measurement tool" in out.lower()  # disclaimer is always printed


def test_cli_not_deployable_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(123)
    returns = 0.01 * rng.standard_normal(1500)
    csv = tmp_path / "noise.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])

    code = main([str(csv), "--n-trials", "1"])
    out = capsys.readouterr().out
    assert code == 1
    assert "NOT_DEPLOYABLE" in out


def test_cli_matrix_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(2024)
    candidates = 0.01 * rng.standard_normal((600, 8))
    csv = tmp_path / "matrix.csv"
    _write_returns(csv, candidates, [f"cfg{i}" for i in range(8)])

    code = main([str(csv)])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "PBO" in out


def test_cli_column_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(42)
    good = 0.0008 + 0.008 * rng.standard_normal(1500)
    noise = 0.01 * rng.standard_normal(1500)
    csv = tmp_path / "two.csv"
    _write_returns(csv, np.column_stack([good, noise]), ["good", "noise"])

    code = main([str(csv), "--column", "good", "--n-trials", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "DEPLOYABLE" in out


def test_cli_about(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--about"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Bailey" in out
    assert "Deflated Sharpe" in out


def test_cli_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["does_not_exist_12345.csv"])
    err = capsys.readouterr().err
    assert code == 2
    assert "error" in err.lower()


def test_cli_requires_path_or_about(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_writes_html_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(4)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    csv = tmp_path / "good.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])
    report = tmp_path / "verdict.html"

    code = main([str(csv), "--n-trials", "1", "--report", str(report)])
    captured = capsys.readouterr()
    assert code == 0
    assert report.exists()
    assert report.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    # The "written" confirmation goes to stderr so stdout stays parseable.
    assert "Report written to" in captured.err


def test_cli_writes_markdown_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    csv = tmp_path / "good.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])
    report = tmp_path / "verdict.md"

    code = main([str(csv), "--n-trials", "1", "--report", str(report)])
    assert code == 0
    assert report.read_text(encoding="utf-8").startswith("# Lyra Validate")


def test_cli_json_output_is_valid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(4)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    csv = tmp_path / "good.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])

    code = main([str(csv), "--n-trials", "1", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)  # must be parseable JSON, nothing else on stdout
    assert payload["classification"] == "DEPLOYABLE"
    assert payload["deployable"] is True
    assert payload["pbo"] is None  # NaN -> null for a single strategy
    assert "disclaimer" in payload
    # Evidence-gap fields: MinTRL is finite and met here; a single trial has no
    # multiple-testing minimum backtest length.
    assert 0 < payload["min_track_record"] <= payload["n_periods"]
    assert payload["min_backtest_years"] == 0.0


def test_cli_text_output_reports_min_trl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(4)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    csv = tmp_path / "good.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])

    code = main([str(csv), "--n-trials", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "MinTRL" in out


def test_cli_trials_alias(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    csv = tmp_path / "good.csv"
    _write_returns(csv, returns.reshape(-1, 1), ["strategy"])
    assert main([str(csv), "--trials", "1"]) == 0


def test_cli_on_packaged_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _SAMPLE_CSV.exists(), "examples/sample_returns.csv should ship with the package"
    report = tmp_path / "sample.html"
    code = main([str(_SAMPLE_CSV), "--report", str(report)])
    out = capsys.readouterr().out
    assert code == 0
    assert "DEPLOYABLE" in out
    assert report.exists()
