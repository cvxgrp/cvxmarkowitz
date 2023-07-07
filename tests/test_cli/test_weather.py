# -*- coding: utf-8 -*-
from click.testing import CliRunner
from mock import patch
from requests import Response

from cvx.cli.weather import cli


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["temperature"])
    # print(result.output)
    assert result.exit_code == 0
    assert -90.0 <= float(result.output) <= 60.0


def test_unknown_metric():
    runner = CliRunner()
    result = runner.invoke(cli, ["XXX"])
    assert result.output == "Metric not supported!\n"


def test_unsuccessful_request():
    runner = CliRunner()
    # Mock a failed request to the API
    r = Response()
    r.status_code = 500

    with patch("requests.get", return_value=r):
        result = runner.invoke(cli, ["temperature"])
        assert result.output.strip() == "Open-Meteo is down!"
