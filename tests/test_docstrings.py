"""Tests for module docstrings using doctest.

Automatically discovers all packages under `src/`
and runs doctests for each.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root, asserting pyproject.toml exists.

    This fixture locates the project root by taking the parent directory of the
    tests folder and ensures a pyproject.toml file is present there. It is used
    by other fixtures to resolve package paths.
    """
    root = Path(__file__).parent.parent
    assert (root / "pyproject.toml").is_file()
    return root


@pytest.fixture(scope="session")
def package_paths(project_root: Path) -> list[Path]:
    """Return a list of package directories defined in pyproject.toml."""
    toml_file = project_root / "pyproject.toml"
    with toml_file.open("rb") as f:
        data = tomllib.load(f)

    packages = (
        data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", [])
    )

    return [(project_root / pkg) for pkg in packages]


def _iter_modules_from_path(package_path: Path) -> Iterator[ModuleType]:
    """Yield imported modules recursively from a package path."""
    pkg_root = str(package_path.parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    package_name = package_path.name

    # Walk through package via pkgutil
    for finder, name, ispkg in pkgutil.walk_packages([str(package_path)], prefix=f"{package_name}."):
        try:
            module = importlib.import_module(name)
            yield module
        except Exception as exc:
            warnings.warn(f"Could not import {name}: {exc}", stacklevel=1)


def test_toml_file(package_paths):
    """Run doctests for all configured packages from pyproject.toml.

    The test iterates through each package path discovered in the project
    configuration and invokes doctest on all importable modules within.
    """
    for path in package_paths:
        run_doctests_for_package(path)


def run_doctests_for_package(package_path: Path) -> None:
    """Run doctests on all modules in a package."""
    modules = list(_iter_modules_from_path(package_path))

    total_tests = 0
    total_failures = 0
    failed_modules = []

    for module in modules:
        results = doctest.testmod(
            module,
            verbose=False,
            optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE),
        )
        total_tests += results.attempted

        if results.failed:
            total_failures += results.failed
            failed_modules.append((module.__name__, results.failed, results.attempted))

    if failed_modules:
        formatted = "\n".join(f"  {name}: {failed}/{attempted} failed" for name, failed, attempted in failed_modules)
        msg = (
            f"Doctest summary: {total_tests} tests across {len(modules)} modules\n"
            f"Failures: {total_failures}\n"
            f"Failed modules:\n{formatted}"
        )
        assert total_failures == 0, msg

    if total_tests == 0:
        warnings.warn(f"No doctests were found in package {package_path.name}", stacklevel=1)
