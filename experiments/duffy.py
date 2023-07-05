# -*- coding: utf-8 -*-
import cvxpy as cp

if __name__ == "__main__":
    x = cp.Variable(2, "x")
    objective = cp.Minimize(cp.sum_squares(x) + cp.abs(x[0] - 1) + cp.abs(x[1] - x[0]))
    problem = cp.Problem(objective)
    problem.solve()
    print(problem.status)
    print(x.value)
