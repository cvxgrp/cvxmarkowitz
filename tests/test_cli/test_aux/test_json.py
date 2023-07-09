# -*- coding: utf-8 -*-
import numpy as np

from cvx.cli.aux.json import read_json, write_json


def test_read_and_write_json(tmp_path):
    data = {"a": np.array([2.0, 3.0]), "b": 3.0, "c": "test"}
    write_json(tmp_path / "test.json", data)

    recovered_data = dict(read_json(tmp_path / "test.json"))

    assert data["b"] == recovered_data["b"]
    assert data["c"] == recovered_data["c"]

    # check numpy arrays are equal
    np.testing.assert_array_equal(data["a"], recovered_data["a"])
