# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Base:
    foo: int


@dataclass(frozen=True)
class Child(Base):
    fooChild: int = None

    def __post_init__(self):
        object.__setattr__(self, "fooChild", self.foo)


if __name__ == "__main__":
    a = Base(foo=10)
    print(a.foo)

    c = Child(foo=10)
    print(c.fooChild)
