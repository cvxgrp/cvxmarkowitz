# -*- coding: utf-8 -*-
import json
from collections.abc import Iterable

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_json(json_file):
    """Read a json file and return a generator of key-value pairs"""
    with open(json_file, "r") as f:
        json_data = json.load(f)
        for name, data in json_data.items():
            if isinstance(data, Iterable):
                yield name, np.asarray(data)
            else:
                yield name, data


def write_json(json_file, data):
    """Write a json file with the data"""
    with open(json_file, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder)
