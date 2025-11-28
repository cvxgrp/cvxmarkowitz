"""Small CVXPY demo inspired by a Daniel Duffy post."""

# see https://www.linkedin.com/posts/daniel-j-duffy-a6ab3912_fxy-x-x-y-y-stdabsx-1-ugcPost-7082323397543632896-SFZ1/?utm_source=share&utm_medium=member_desktop
import cvxpy as cp

if __name__ == "__main__":
    x = cp.Variable(2, "x")
    objective = cp.Minimize(cp.sum_squares(x) + cp.abs(x[0] - 1) + cp.abs(x[1] - x[0]))
    problem = cp.Problem(objective)
    problem.solve()
    print(problem.status)
    print(x.value)
