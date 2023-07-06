# -*- coding: utf-8 -*-
from click.testing import CliRunner

from cvx.cli.weather import cli


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["temperature"])
    assert result.exit_code == 0
    assert -90.0 <= float(result.output) <= 60.0
