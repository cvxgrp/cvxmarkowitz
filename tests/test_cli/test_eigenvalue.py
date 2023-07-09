# -*- coding: utf-8 -*-
import json

import numpy as np
import pytest
from click.testing import CliRunner

from cvx.cli.smallest_ev import smallest_ev
from cvx.linalg.random import rand_cov


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def test_cli(resource_dir):
    runner = CliRunner()
    result = runner.invoke(smallest_ev, [str(resource_dir / "eigenvalue.json")])
    assert result.exit_code == 0
    assert float(result.output) == pytest.approx(0.13132462643944098)


def test_large():
    runner = CliRunner()
    with runner.isolated_filesystem():
        matrix = rand_cov(1000)
        with open("eigenvalue.json", "w") as f:
            json.dump({"a": matrix}, f, cls=NumpyEncoder)

        result = runner.invoke(smallest_ev, ["eigenvalue.json"])
        assert result.exit_code == 0
        assert float(result.output) >= 0


def test_file_not_found():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Run the CLI command with a non-existent file
        result = runner.invoke(smallest_ev, ["nonexistent_file.txt"])

        assert result.exit_code != 0
        assert (
            result.output.strip()
            == "Usage: smallest-ev [OPTIONS] JSON_FILE\nTry 'smallest-ev --help' for help.\n\n"
            "Error: Invalid value for 'JSON_FILE': File 'nonexistent_file.txt' does not exist."
        )
