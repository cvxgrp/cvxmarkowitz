# -*- coding: utf-8 -*-
import json

import click
import numpy as np


@click.command()
@click.argument("json_file", type=click.Path(exists=True, dir_okay=False))
def smallest_ev(json_file) -> None:
    """Print FILENAME if the file exists."""
    with open(json_file, "r") as f:
        json_data = json.load(f)

        a_restored = np.asarray(json_data["a"])
        w, v = np.linalg.eigh(a_restored)
        idx = w.argsort()
        w = w[idx]
        print(w[0])


# if __name__ == "__main__":
#    smallest_ev()
