import numpy as np
from ortools.linear_solver import pywraplp


c = np.loadtxt("c.csv", delimiter="\t")
od_wgt = np.loadtxt("od_wgt.csv", delimiter=",")
max_wgt = np.loadtxt("max_wgt.csv", delimiter=",")
od1 = np.loadtxt("od1.csv", delimiter=",")
od2 = np.loadtxt("od2.csv", delimiter=",")

num_i, num_j = c.shape
Inf = 1e8
loading_rate = 0.5

# solver
solver = pywraplp.Solver('cbc_mip', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
# solver = pywraplp.Solver('glpk_mip', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)  # something wrong

x =[
    [
        solver.IntVar(0, Inf, "x%d" % (10*i+j))
        for j in range(num_j)
    ]
    for i in range(num_i)
]