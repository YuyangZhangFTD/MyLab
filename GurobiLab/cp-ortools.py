import numpy as np
from collections import defaultdict
from ortools.constraint_solver import pywrapcp

example = 1

if example == 1:
    # example 1, optimal 350
    num_s = 4
    num_d = 3
    supply = [10, 30, 40, 20]
    demand = [20, 50, 30]
    c = np.array([
        [2, 3, 4],
        [3, 2, 1],
        [1, 4, 3],
        [4, 5, 2]
    ])
    f = np.array(
        [[10, 30, 20] for _ in range(4)]
    )
    bar_y = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0]
    ])
elif example == 2:
    # example 2, optimal 4541
    num_s = 8
    num_d = 7
    supply = [20, 20, 20, 18, 18, 17, 17, 10]
    demand = [20, 19, 19, 18, 17, 16, 16]
    c = np.array([
        [31, 27, 28, 10, 7, 26, 30],
        [15, 19, 17, 7, 22, 17, 16],
        [21, 17, 19, 29, 27, 22, 13],
        [9, 19, 7, 15, 20, 17, 22],
        [19, 7, 18, 10, 12, 27, 23],
        [8, 16, 10, 10, 11, 13, 15],
        [14, 32, 22, 10, 22, 15, 19],
        [30, 27, 24, 26, 25, 15, 19]
    ])
    f = np.array([
        [649, 685, 538, 791, 613, 205, 467],
        [798, 211, 701, 506, 431, 907, 945],
        [687, 261, 444, 264, 443, 946, 372],
        [335, 385, 967, 263, 423, 592, 939],
        [819, 340, 233, 889, 211, 854, 823],
        [307, 620, 845, 919, 223, 854, 823],
        [560, 959, 782, 417, 358, 589, 383],
        [375, 791, 720, 416, 251, 887, 235]
    ])
    bar_y = np.array([
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 0]
    ])
else:
    print("Example not found")

M = np.ones((num_s, num_d))
for i in range(num_s):
    for j in range(num_d):
        M[i, j] = min(supply[i], demand[j])

solver = pywrapcp.Solver("cp")

x = dict()
y = dict()
for i in range(num_s):
    for j in range(num_d):
        x[i, j] = solver.NumVar(0, solver.infinity(), 'x[%i,%i]' % (i, j))
        y[i, j] = solver.IntVar(0, 1, 'y[%i,%i]' % (i, j))

# add constraints
u = dict()
for i in range(num_s):
    u[i] = solver.Add(solver.Sum([x[i, j] for j in range(num_d)]) <= supply[i])
v = dict()
for j in range(num_d):
    v[i] = solver.Add(solver.Sum([x[i, j] for i in range(num_s)]) >= demand[j])
w = dict()
for i in range(num_s):
    for j in range(num_d):
        w[i, j] = solver.Add(x[i, j] <= M[i, j] * y[i, j])

obj = solver.Minimize(solver.Sum([
    solver.Sum([
        c[i, j] * x[i, j] + f[i, j] * y[i, j]
        for i in range(num_s)
    ]) for j in range(num_d)
]), 1)

db = solver.Phase(x, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)