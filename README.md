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

>*Before Markowitz, mathematical analysis of investing was limited
to Graham & Dodd-type analysis of individual stocks and John Burr Williams’
Theory of Investment Value, also focused on individual investments.
Markowitz provided a precise mathematical definition of risk as standard deviation of return,
and focused on return and risk at the portfolio level, leading to modern portfolio theory
and much of the portfolio analytics we run today.
Like many profound breakthroughs, it looks obvious in retrospect.*
-- <cite>[[Ronald Kahn](https://en.wikipedia.org/wiki/Ronald_Kahn) in 2023]</cite>

Our assumption is that we solve multiple problems of the same type in a row. The input for the $$n$$th problem may depend
on the outcome of a previous problem, e.g. the $$n-1$$th.
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
