# -*- coding: utf-8 -*-
from click.testing import CliRunner

from cvx.cli.minvariance import minvariance


def test_cli(resource_dir):
    runner = CliRunner()
    result = runner.invoke(minvariance, [str(resource_dir / "matrix.json")])
    assert result.exit_code == 0
    assert "Solution" in result.output


def test_cli_serialize(resource_dir, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        minvariance, [str(resource_dir / "matrix.json"), str(tmp_path / "problem.pkl")]
    )
    assert result.exit_code == 0
    assert "Solution" in result.output

    result = runner.invoke(
        minvariance, [str(resource_dir / "matrix.json"), str(tmp_path / "problem.pkl")]
    )
    assert result.exit_code == 0
    assert "Solution" in result.output


def test_options(resource_dir):
    runner = CliRunner()
    result = runner.invoke(
        minvariance, [str(resource_dir / "matrix.json"), "--assets", "20"]
    )

    assert result.exit_code == 0
    assert "Solution" in result.output


def test_infeasible(resource_dir):
    runner = CliRunner()
    result = runner.invoke(minvariance, [str(resource_dir / "matrix_infeasible.json")])

    assert result.exit_code > 0
