# -*- coding: utf-8 -*-
import traceback

import fire

from cvx.cli.aux.io import exists
from cvx.cli.aux.json import read_json
from cvx.markowitz.builder import deserialize
from cvx.markowitz.portfolios.min_var import MinVar, estimate_dimensions


def cli(json_file, problem_file=None, assets=None, factors=None) -> None:
    # parse the json file with input data for the problem
    try:
        input_data = dict(read_json(json_file))

        does_exist = exists(problem_file)

        if does_exist:
            # the problem has been serialized before and we can reuse it
            problem = deserialize(problem_file)

        else:
            # build the problem from scratch
            if assets is not None:
                # the user has specified the number of assets and factors
                # useful as the user can specify coarse upper bounds
                problem = MinVar(assets=assets, factors=factors).build()
            else:
                assets, factors = estimate_dimensions(**input_data)
                print(
                    f"Estimated the numbers of assets as {assets} and factors as {factors}"
                )
                problem = MinVar(assets=assets, factors=factors).build()

            # We have constructed the problem, write it to file if the file has been specified
            if problem_file is not None:
                print(f"Serializing problem in {problem_file}")
                problem.serialize(problem_file)

        problem.update(**input_data)
        problem.solve()
        return problem.weights

    except Exception as e:
        print(traceback.print_exception(type(e), e, e.__traceback__))
        raise e


def main():  # pragma: no cover
    """
    Run the CLI using Fire
    """
    fire.Fire(cli)
