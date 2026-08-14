# [cvxmarkowitz](http://www.cvxgrp.org/cvxmarkowitz)

[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxmarkowitz/blob/main/LICENSE)
[![Coverage](https://www.cvxgrp.org/cvxmarkowitz/coverage-badge.svg)](https://www.cvxgrp.org/cvxmarkowitz/reports/html-coverage/)

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
and implements the abstract property [objective](src/cvxmarkowitz/builder.py#L92).
The builder remains flexible. At this stage it is possible to add or remove
constraints. Only once we trigger the build() method do we construct
the problem and compile it.

For injecting values for data and parameters into the problem,
we use the [update](src/cvxmarkowitz/problem.py) method. It overwrites the
parameter values **in place** and returns `None` — there is only ever one
problem, which is precisely what lets CVXPY reuse the cached compilation.

The builder picks a risk model for you: a `FactorModel` when you pass `factors`,
a `SampleCovariance` otherwise. To use a different one — `CVar`, say — pass it in
under `ModelName.RISK` and the builder keeps yours instead of defaulting:

```python
from cvxmarkowitz import MinVar
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.risk import CVar

builder = MinVar(assets=14, model={M.RISK: CVar(alpha=0.95, rows=50, assets=14)})
print(type(builder.risk).__name__)
```

```result
CVar
```

## Installation

The package is not published on PyPI. Install it from the git source:

```bash
pip install git+https://github.com/cvxgrp/cvxmarkowitz
```

## Usage

Build a minimum-variance problem once, then re-solve it repeatedly with new
data. The compiled problem is [DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)-compliant,
so subsequent solves reuse the cached compilation.

```python
import numpy as np
from cvx.linalg import cholesky

from cvxmarkowitz import MinVar
from cvxmarkowitz.names import DataNames as D

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

### Reusing a built problem

The problems are parameterized (DPP-compliant), so a single built problem can be
re-solved with fresh data. Build once, then `update` and `solve` in a loop — the
cvxpy canonicalization is paid for on the first solve and reused afterwards.

```python
problem = MinVar(assets=4).build()  # build once

for correlation in (0.0, 0.5, 0.9):
    problem.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, correlation], [correlation, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.zeros(2),
        }
    )
    print(f"rho={correlation}: objective={problem.solve():.4f}")
```

```result
rho=0.0: objective=0.8165
rho=0.5: objective=0.9354
rho=0.9: objective=0.9958
```

## Errors

Everything the package raises derives from `CvxError`, so a single
`except CvxError` still catches all of it. The subclasses below separate the
failure modes that want different handling:

| Error | Raised when | Retry helps? |
|---|---|---|
| `CvxDataError` | required data is missing, or shapes disagree | yes, with corrected input |
| `CvxBuildError` | the assembled problem is not DPP-compliant | no — the formulation has to change |
| `CvxSolverError` | the solver returned a non-optimal status | no — try another solver or relax the problem |

The `CvxBuildError` check is a raise rather than an `assert`, so it still fires
under `python -O`.

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

## experiments

`experiments/` holds standalone research scripts — backtests and the figures
behind the talks — not part of the installed package. They pull in the `dev`
dependency group (`yfinance`, `loguru`, `cvxsimulator`, `tinycta`, `plotly`),
so run them against the full development environment:

```bash
make install
uv run --group dev python experiments/minRisk1.py
```

They are deliberately outside the quality gates: `make typecheck`,
`make docs-coverage`, `make deptry` and `make security` all scope to `src/`, and
the coverage gate scopes to `tests/`. Only `make fmt` reaches them, since
pre-commit runs repo-wide. Treat them as scratch work — if something here earns
a stability guarantee, it belongs in `src/cvxmarkowitz/`.
