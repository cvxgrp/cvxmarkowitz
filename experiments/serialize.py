# -*- coding: utf-8 -*-
import pickle

from cvx.markowitz.portfolios.min_var import MinVar

if __name__ == "__main__":
    builder = MinVar(assets=20, factors=10)
    problem = builder.build()

    with open("test.pickle", "wb") as outfile:
        # "wb" argument opens the file in binary mode
        pickle.dump(problem, outfile)
