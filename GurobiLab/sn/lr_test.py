import numpy as np
from gurobi import *


n, m = 40, 20

A = np.loadtxt("A.txt")
b = np.loadtxt("b.txt")
c = np.loadtxt("c.txt")

mip = Model()

x = mip.addVars([j for j in range(m)], vtype=GRB.CONTINUOUS, name="x")
q = mip.addVars([j for j in range(10)], vtype=GRB.INTEGER, name="q")

mip.setObjective(quicksum(c[j]*x[j] for j in range(m)), GRB.MINIMIZE)

mip.addConstrs((
    quicksum(A[i,j] * x[j] for j in range(m)) >= b[i] 
    for i in range(10, n)
))

mip.addConstrs((
    quicksum(A[i,j] * x[j] for j in range(m)) >= b[i] * q[i]
    for i in range(10)
))

mip.addConstr(
    quicksum(q[i] for i in range(10)) >= 5
)

mip.optimize()