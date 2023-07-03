# [cvxmarkowitz](http://www.cvxgrp.org/cvxmarkowitz/)

[![PyPI version](https://badge.fury.io/py/cvxmarkowitz.svg)](https://badge.fury.io/py/cvxmarkowitz)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/simulator/blob/master/LICENSE)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxmarkowitz.svg)](https://pypi.python.org/pypi/cvxmarkowitz/)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/simulator/badge.png?branch=main)](https://coveralls.io/github/cvxgrp/simulator?branch=main)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxmarkowitz)

## Motivation

We stand on the shoulders of [CVXPY](https://www.cvxpy.org).

We solve problems arising in portfolio construction following the ideas of [Harry Markowitz](https://en.wikipedia.org/wiki/Harry_Markowitz).
Markowitz gave diversification a mathematical home in the 1950s.

Our assumption is that we solve multiple problems of the same type in a row. The input for the $n$th problem may depend
on the outcome of a previous problem, e.g. the $n-1$th. Hence, we need to respect their sequential nature and order.

We can however hope that the problems we construct are [DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)
compliant. The first time a DPP-compliant problem is solved, CVXPY compiles it and caches the mapping from parameters to problem data. As a result, subsequent rewritings of DPP problems can be substantially faster.

In practice, the problems are not constant in size. Assets are added or removed, factors are added or removed, and so on.
We expect the user is providing the number of assets a priori.
We can then construct a problem suitable for a number of assets equal or smaller than the one provided.
Using this approach, we keep the number of assets fixed by setting the weights for the assets not used to zero.
Hence we do **not** need to recompile the problem as a new asset has to be added.

Every problem has be constructed by a Builder. Here's a builder for a classic [minimum variance problem](cvx/markowitz/portfolios/min_var.py).
The builder inherits from the [Builder](cvxmarkowitz/markowitz/builder.py) and implements the abstract method [build](cvxmarkowitz/markowitz/builder.py#L15).
The builder remains flexible. At this stage it is possible to add or recome constraints,  Only once we trigger the build() method, we construct
the problem and compile it.

For injecting values for data and parameter into the problem, we use the [update](cvxmarkowitz/markowitz/builder.py#L19) method.





## Installation

You can install the package via [PyPI](https://pypi.org/project/cvxmarkowitz/):

```bash
pip install cvxmarkowitz
```


## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
poetry install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml) and fixed in [poetry.lock](poetry.lock).

## Kernel

We install [JupyterLab](https://jupyter.org) within your new virtual environment. Executing

```bash
./create_kernel.sh
```

constructs a dedicated [Kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) for the project.
