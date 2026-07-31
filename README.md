# [cvxmarkowitz](http://www.cvxgrp.org/cvxmarkowitz/book)

[![PyPI version](https://badge.fury.io/py/cvxmarkowitz.svg)](https://badge.fury.io/py/cvxmarkowitz)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxmarkowitz/blob/master/LICENSE)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxmarkowitz.svg)](https://pypi.python.org/pypi/cvxmarkowitz/)
[![Coverage](https://www.cvxgrp.org/cvxmarkowitz/coverage-badge.svg)](https://www.cvxgrp.org/cvxmarkowitz/reports/html-coverage/)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxmarkowitz)

## Motivation

We stand on the shoulders of [CVXPY](https://www.cvxpy.org).

We solve problems arising in portfolio construction following the ideas of
[Harry Markowitz](https://en.wikipedia.org/wiki/Harry_Markowitz). Markowitz gave
diversification a mathematical home in the 1950s.

Our assumption is that we solve multiple problems of the same type in a row.
The input for the $n$th problem may depend on the outcome of a previous problem,
e.g. the $n-1$th. Hence, we need to respect their sequential nature and order.

We can however hope that the problems we construct are [DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)
compliant. The first time a DPP-compliant problem is solved, CVXPY compiles it
and caches the mapping from parameters to problem data. As a result, subsequent
rewritings of DPP problems can be substantially faster.

In practice, the problems are not constant in size. Assets are added or removed,
factors are added or removed, and so on. We expect the user is providing the
number of assets a priori. We can then construct a problem suitable for a number
of assets equal or smaller than the one provided. Using this approach, we keep
the number of assets fixed by setting the weights for the assets not used to
zero. Hence we do **not** need to recompile the problem as a new asset has to be
added.

Every problem has to be constructed by a Builder. Here's a builder for a classic
[minimum variance problem](src/cvxmarkowitz/portfolios/min_var.py).
The builder inherits from the [Builder](src/cvxmarkowitz/builder.py)
and implements the abstract method [build](src/cvxmarkowitz/builder.py#L95).
The builder remains flexible. At this stage it is possible to add or remove
constraints. Only once we trigger the build() method do we construct
the problem and compile it.

For injecting values for data and parameters into the problem,
we use the [update](src/cvxmarkowitz/problem.py#L84) method.

## Installation

You can install the package via [PyPI](https://pypi.org/project/cvxmarkowitz/):

```bash
pip install cvxmarkowitz
```

## Usage

Build a minimum-variance problem once, then re-solve it repeatedly with new
data. The compiled problem is [DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)-compliant,
so subsequent solves reuse the cached compilation.

```python
import numpy as np
from cvx.linalg import cholesky

from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.portfolios.min_var import MinVar

# Build a long-only, budget-constrained minimum-variance problem for 4 assets.
problem = MinVar(assets=4).build()

# Inject data and parameters. Here only 2 of the 4 asset slots are used;
# the unused assets are pinned to zero weight.
problem.update(
    **{
        D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        D.LOWER_BOUND_ASSETS: np.zeros(2),
        D.UPPER_BOUND_ASSETS: np.ones(2),
        D.VOLA_UNCERTAINTY: np.zeros(2),
    }
)

objective = problem.solve()  # defaults to the CLARABEL solver

print("objective:", round(objective, 4))
print("weights:", np.round(problem.weights, 3))
```

```result
objective: 0.9354
weights: [0.75 0.25 0.   0.  ]
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) and a
[Rhiza](https://github.com/jebel-quant/rhiza)-managed `Makefile`. To create the
virtual environment defined in `pyproject.toml` and locked in `uv.lock`:

```bash
make install
```

## marimo

We install [marimo](https://marimo.io) on the fly within the virtual
environment. Executing

```bash
make marimo
```

will install and start marimo.
