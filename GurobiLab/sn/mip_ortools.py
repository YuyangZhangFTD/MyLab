import numpy as np
from ortools.linear_solver import pywraplp
import time

c = np.loadtxt("c.csv", delimiter="\t")
od_wgt = np.loadtxt("od_wgt.csv", delimiter=",")
max_wgt = np.loadtxt("max_wgt.csv", delimiter=",")
od1 = np.loadtxt("od1.csv", delimiter=",")
od2 = np.loadtxt("od2.csv", delimiter=",")

num_i, num_j = c.shape
M = 1e8
loading_rate = [0.5 for _ in range(num_j)]
l = 3

# num_j = 250

# init solver
solver = pywraplp.Solver('cbc_mip', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# add variables
x = dict()
q = dict()
for j in range(num_j):
    q[j] = solver.IntVar(0, l, "q[%i]" % j)
    for i in range(num_i):
        x[i, j] = solver.NumVar(0, M, 'x[%i,%i]' % (i, j))

# add objective
z = solver.Sum([
    x[i, j]
    for i in range(num_i)
    for j in range(num_j)
])
solver.Maximize(z)

# add Constraints
for i in range(num_i):
    for j in range(num_j):
        solver.Add(x[i, j] <= c[i, j] * M)

for i in range(num_i):
    solver.Add(solver.Sum([
        x[i, j]
        for j in range(num_j)
    ]) <= od_wgt[i])

for j in range(num_j):
    solver.Add(solver.Sum([
        x[i, j] for i in od1[j, :]
    ]) <= max_wgt[j] * q[j])

    solver.Add(solver.Sum([
        x[i, j] for i in od2[j, :]
    ]) <= max_wgt[j] * q[j])

    solver.Add(solver.Sum([
        x[i, j] for i in od1[j, :]
    ]) >= max_wgt[j] * (q[j] - 1 + loading_rate[j]))

    solver.Add(solver.Sum([
        x[i, j] for i in od2[j, :]
    ]) >= max_wgt[j] * (q[j] - 1 + loading_rate[j]))

start = time.clock()
# optimize
solver.Solve()
end = time.clock()

print('z = ', solver.Objective().Value())
print(end - start)
