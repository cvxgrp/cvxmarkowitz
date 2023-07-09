# -*- coding: utf-8 -*-
from cvx.cli.aux.io import exists


def test_exists_none():
    assert not exists()


def test_with_file_existing(resource_dir):
    assert exists(resource_dir / "stock_prices.csv")


def test_with_file_not_exisiting(resource_dir):
    assert not exists(resource_dir / "not_existing.csv")
