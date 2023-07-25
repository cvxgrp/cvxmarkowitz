# -*- coding: utf-8 -*-
import pytest

from cvx.cli.minvariance import cli as minvariance
from cvx.markowitz.cvxerror import CvxError


def test_cli(resource_dir):
    minvariance(json_file=resource_dir / "matrix.json")


def test_cli_serialize(resource_dir, tmp_path):
    minvariance(
        json_file=resource_dir / "matrix.json", problem_file=tmp_path / "problem.pkl"
    )

    minvariance(resource_dir / "matrix.json", tmp_path / "problem.pkl")


def test_options(resource_dir):
    minvariance(resource_dir / "matrix.json", assets=20)


def test_infeasible(resource_dir):
    with pytest.raises(CvxError):
        minvariance(resource_dir / "matrix_infeasible.json")
