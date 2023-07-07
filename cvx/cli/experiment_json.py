# -*- coding: utf-8 -*-
# https://stackoverflow.com/a/47626762/1695486
import json
from pathlib import Path

import numpy as np
from aux.rand_cov import rand_cov


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)
json_dump = json.dumps({"a": a, "b": 2 * a}, cls=NumpyEncoder)
# print(json_dump)

# json_load = json.loads(json_dump)
# a_restored = np.asarray(json_load["a"])
# print(a_restored)
# print(a_restored.shape)

# output to file
path = Path(__file__).parent / "data"
print(json_dump)

with open(path / "ex2.json", "w") as f:
    json.dump({"a": a, "b": 2 * a}, f, cls=NumpyEncoder)


with open(path / "ex2.json", "r") as f:
    json_data = f.read()
    json_load = json.loads(json_data)
    print(type(json_load))
    print(json_load)
    print(json_load.keys())
    a_restored = np.asarray(json_load["a"])
    print(a_restored)
    print(a_restored.shape)

a = rand_cov(5)
with open(path / "eigenvalue.json", "w") as f:
    json.dump({"a": a}, f, cls=NumpyEncoder)
