# CLAUDE.md

Guidance for working in this repository.

## What this is

`cvxmarkowitz` builds [Markowitz](https://en.wikipedia.org/wiki/Harry_Markowitz)
portfolio-construction problems on top of [cvxpy](https://www.cvxpy.org). The
design goal is stated in `README.md` and shapes everything else: a problem is
**compiled once and re-solved many times** with fresh data. That requires every
assembled problem to be
[DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)-compliant,
so cvxpy can cache its canonicalization across solves.

Two consequences that are easy to violate by accident:

- **`Problem.update` mutates in place and returns `None`.** There is only ever
  one problem; a second `update` overwrites the first. `Problem` is
  `frozen=True`, which stops attribute rebinding but deliberately not mutation
  of the cvxpy Parameters it holds. Do not change this to copy semantics.
- **`Builder.build` raises `CvxBuildError` on a non-DPP problem — a `raise`, not
  an `assert`.** `assert` is stripped under `python -O`, and DPP compliance is
  the invariant the whole caching story rests on.
  `tests/test_markowitz/test_builder.py` pins this by running the check in a
  real `python -O` subprocess.

## Repository layout and ownership

This repo is **rhiza-managed**: `.rhiza/template.yml` points at
`jebel-quant/rhiza` (currently `v1.3.3`) and a large part of the development
infrastructure is synced from there. Knowing which half you are editing matters,
because a change to a template-owned file is reverted by the next
`/rhiza:update`.

**Locally owned — edit freely:**

| Path | What |
| --- | --- |
| `src/cvxmarkowitz/` | the library |
| `tests/` | the test suite (except `tests/test_rhiza_packaging.py`) |
| `experiments/` | standalone research scripts, not part of the package |
| `pyproject.toml` | manifest, tool config, dependency groups |
| `README.md`, `mkdocs.yml` | project docs and site nav |
| `.rhiza/template.yml` | which template ref this repo tracks |
| `.clusterfuzzlite/`, `copyright.txt`, `portfolio.png` | local extras |

**Template-owned — fix upstream, not here:** `.github/**`, everything under
`.rhiza/` except `template.yml`, all of `docs/`, plus `ruff.toml`, `pytest.ini`,
`.pre-commit-config.yaml`, `.bandit`, `.editorconfig`, `.gitignore`,
`.python-version`, `cliff.toml`, `LICENSE`, `SECURITY.md` and
`tests/test_rhiza_packaging.py`.

The authoritative list is the `files:` block of `.rhiza/template.lock` — 68
paths, machine-generated, do not hand-edit.

**The documented extension points** are the exception to the rule above. These
are template-delivered but merged rather than replaced, so local edits survive a
sync:

- `Makefile` — a thin shim that sets overrides and then `include`s
  `.rhiza/rhiza.mk`. This is where `COVERAGE_FAIL_UNDER = 100` lives, raising the
  template's default of 90. Keep it small.
- `.rhiza/make.d/custom-task.mk` and `custom-env.mk` — repo-specific targets.
- `local.mk` — developer-local, gitignored, optional.

## Commands

Always go through `make`; the flags, thresholds and exclusions live in the
targets, so invoking the tools directly measures something else.

| Command | What it runs |
| --- | --- |
| `make install` | create `.venv` from `uv.lock`, install pre-commit hooks |
| `make test` | pytest with the 100% coverage gate |
| `make fmt` | the full pre-commit suite (ruff, markdownlint, bandit, actionlint, …) |
| `make typecheck` | `ty` **and** `mypy --strict` over `src/` |
| `make docs-coverage` | interrogate, at a 100% minimum |
| `make deps` | deptry (`make deptry` is the deprecated spelling) |
| `make security` | bandit |
| `make rhiza-test` | the template's own bundled tests under `.rhiza/tests/` |
| `make all` | everything CI runs |
| `make help` | the full target list |

## Conventions

**Both type checkers matter.** `pyproject.toml` treats cvxpy as untyped for
mypy, because its functional atoms are exported dynamically and `mypy --strict`
reports spurious errors at every call site. The cost is that every cvxpy symbol
is `Any` to mypy — so mypy's "no issues found" covers the dataclass plumbing and
the numpy edges, *not* the optimisation semantics. `ty` covers those. Do not
drop `ty` from `make typecheck` on the grounds that mypy passes.

**Use the constants in `names.py`.** `DataNames`, `ModelName`, `ConstraintName`
and `ParameterName` exist so that a key is spelled once. Indexing `data`,
`parameter` or `kwargs` with a string literal that has a constant is a bug even
when the strings happen to agree today: presence checks iterate the constants,
so a later rename splits the two paths silently.

**A model must declare every keyword its `update` consumes.** `Problem.update`
guards inputs by walking `Model.keywords`, which defaults to the keys of
`data`. If a model reads a keyword backed by `parameter` instead — as
`ExpectedReturns` does for `mu_uncertainty` — it **must** override `keywords`,
or a caller who omits that keyword gets a bare `KeyError` rather than a
`CvxDataError`.

**Everything raised derives from `CvxError`.** `CvxDataError`, `CvxBuildError`
and `CvxSolverError` split the failure modes by whether retrying helps; the
table in `README.md` is the contract. `tests/test_markowitz/test_public_api.py`
enforces that no exported exception escapes the tree.

**The risk package owns its own defaults.** `Builder` calls
`cvxmarkowitz.risk.default_risk_model` rather than naming `FactorModel` or
`SampleCovariance` itself, so adding a default is a change to `risk/`, not to
the abstract base class. `Bounds` is still imported directly by `builder.py` on
purpose — it is unconditional structure, not a choice among alternatives.

**Both gates are at 100% and should stay there.** Test coverage
(`COVERAGE_FAIL_UNDER = 100`) and docstring coverage (interrogate). New code
needs tests and docstrings in the same change.

**Tests are grouped by behaviour, not mirrored onto modules.** `tests/` does not
follow the one-test-file-per-source-module convention; the opt-out is recorded
in `[tool.check_test_layout]` in `pyproject.toml`, with per-module coverage
guaranteed by the 100% gate instead. The suite uses no mocks — keep it that way,
and assert behaviour rather than implementation.

**`experiments/` is scratch work.** It sits outside every gate except `make fmt`
(`typecheck`, `docs-coverage`, `deps` and `security` all scope to `src/`). If
something there earns a stability guarantee it belongs in `src/cvxmarkowitz/`.

## Release

Versioning is driven by `bump-my-version`, configured in `pyproject.toml` to
read and rewrite `[project].version` directly. The release workflow commits and
tags itself, so `commit` and `tag` are both `false` there.

The package is **not published to PyPI**. The guard is the literal comment
`# Private :: Do Not Upload` in `pyproject.toml` — `.github/workflows/rhiza_release.yml`
matches it with a plain `grep -R` over the file, so it works despite not being a
real classifier. **Do not delete that line** as tidy-up; it is load-bearing.
