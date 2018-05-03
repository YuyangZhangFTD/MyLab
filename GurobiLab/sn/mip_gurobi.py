import numpy as np
from gurobipy import *


c = np.loadtxt("c.csv", delimiter="\t")
od_wgt = np.loadtxt("od_wgt.csv", delimiter=",")
max_wgt = np.loadtxt("max_wgt.csv", delimiter=",")
od1 = np.loadtxt("od1.csv", delimiter=",", dtype="int")
od2 = np.loadtxt("od2.csv", delimiter=",", dtype="int")

num_i, num_j = c.shape
Inf = 1e8
loading_rate = 0.5

# model
m = Model()

# add variables
x = m.addVars([i for i in range(num_i)],[j for j in range(num_j)], vtype=GRB.CONTINUOUS, name="x")
q = m.addVars([j for j in range(num_j)], vtype=GRB.INTEGER, name="q")

# set objective
m.setObjective(quicksum(x[i,j] for i in range(num_i) for j in range(num_j)), GRB.MAXIMIZE)

# add constraints
# x < cij * inf
m.addConstrs((
    x[i, j] <= c[i,j] * Inf
    for i in range(num_i)
    for j in range(num_j)
))
# \sum_j x <= od_wgt_i
m.addConstrs((
    quicksum(x[i,j] for j in range(num_j)) <= od_wgt[i] 
    for i in range(num_i)
))
# od1
m.addConstrs((
    quicksum(x[i,j] for i in od1[j,:]) <= max_wgt[j] * q[j]
    for j in range(num_j)
))
m.addConstrs((
    quicksum(x[i,j] for i in od1[j,:]) >= max_wgt[j] * (q[j] - 1 + loading_rate)
    for j in range(num_j)
))
# od2
m.addConstrs((
    quicksum(x[i,j] for i in od2[j,:]) <= max_wgt[j] * q[j]
    for j in range(num_j)
))
m.addConstrs((
    quicksum(x[i,j] for i in od2[j,:]) >= max_wgt[j] * (q[j] - 1 + loading_rate)
    for j in range(num_j)
))

m.write("model.mps")

m.optimize()

optx = np.zeros((num_i, num_j))
for i in range(num_i):
    for j in range(num_j):
        optx[i, j] = x[i,j].x if x[i,j].x > 0 else 0

np.savetxt("lp_x.csv", optx, delimiter=",")
