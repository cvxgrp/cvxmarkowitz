# -*- coding: utf-8 -*-
import traceback

import click

from cvx.cli.aux.io import exists
from cvx.cli.aux.json import read_json
from cvx.cli.aux.problem import deserialize_problem, serialize_problem
from cvx.markowitz.portfolios.min_var import MinVar, estimate_dimensions


@click.command()
@click.argument("json_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("problem_file", type=click.Path(dir_okay=False), required=False)
@click.option(
    "--assets",
    "-a",
    default=None,
    type=int,
    required=False,
    help="Number of assets",
)
@click.option(
    "--factors",
    "-f ",
    default=None,
    type=int,
    required=False,
    help="Number of factors",
)
def minvariance(json_file, problem_file=None, assets=None, factors=None) -> None:
    # parse the json file with input data for the problem
    try:
        input_data = dict(read_json(json_file))

        does_exist = exists(problem_file)

        if does_exist:
            # the problem has been serialized before and we can reuse it
            problem = deserialize_problem(problem_file)

        else:
            # build the problem from scratch
            if assets is not None:
                # the user has specified the number of assets and factors
                # useful as the user can specify coarse upper bounds
                problem = MinVar(assets=assets, factors=factors).build()
            else:
                assets, factors = estimate_dimensions(input_data)
                click.echo(
                    f"Estimated the numbers of assets as {assets} and factors as {factors}"
                )
                problem = MinVar(assets=assets, factors=factors).build()

            # We have constructed the problem, write it to file if the file has been specified
            if problem_file is not None:
                click.echo(f"Serializing problem in {problem_file}")
                serialize_problem(problem, problem_file)

        problem.update(**input_data)
        problem.solve()
        click.echo(f"Solution: {problem.weights}")

    except Exception as e:
        click.echo(traceback.print_exception(type(e), e, e.__traceback__))
        raise e
