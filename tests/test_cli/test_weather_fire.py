# -*- coding: utf-8 -*-
import pytest
from mock import patch
from requests import Response

from cvx.cli.weather_fire import cli


def test_cli():
    r = cli("temperature")
    assert -90.0 <= float(r) <= 60.0


def test_unknown_metric():
    with pytest.raises(ValueError):
        cli("XXX")


def test_unsuccessful_request():
    r = Response()
    r.status_code = 500

    with patch("requests.get", return_value=r):
        with pytest.raises(ConnectionError):
            cli("temperature")
