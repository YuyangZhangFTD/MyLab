import numpy as np
from gurobipy import *


def lr_model(x, u, v, w, param):
    # read param
    num_i = param["i"]
    num_j = param["j"]
    num_k = param["k"]
    num_l = param["l"]
    od = param["od"]
    maxwgt = param["maxwgt"]
    lr = param["lr"]
    c = param["c"]
    inf = param["inf"]
    odwgt = param["odwgt"]

    # build model
    m = Model("lr_model")

    # add variables
    x = m.addVars([i for i in range(num_i)],[j for j in range(num_j)], vtype=GRB.CONTINUOUS, name="x")

    # set objective
    m.setObjective(
        quicksum(   # \sum_j
            quicksum(   # \sum_k
                quicksum(   # \sum_i
                    (1 - u[k,j] + v[k,j]) * x[i,j]
                    for i in od[k][j]
                )
                for k in range(num_k)
            )
            for j in range(num_j)
        ), GRB.MAXIMIZE 
    )

    # add constraints
    m.addConstrs((
        x[i,j] <= c[i,j] * inf
        for i in range(num_i)
        for j in range(num_j)
    ))
    m.addConstrs((
        quicksum(
            x[i, j]
            for j in range(num_j)
        ) <= odwgt[i]
        for i in range(num_i)
    ))

    m.optimize()

    primal_x = np.zeros((num_i, num_j))
    if dual.status == GRB.OPTIMAL:
        for i in range(num_i):
            for j in range(num_j):
                primal_x[i,j] = x[i,j].x if x[i,j].x > 0 else 0
    else:
        print("Not optimal: " + str(dual.status))

    return m, primal_x


def dual(x, param):
    # read param
    num_i = param["i"]
    num_j = param["j"]
    num_k = param["k"]
    num_l = param["l"]
    od = param["od"]
    maxwgt = param["maxwgt"]
    lr = param["lr"]
    c = param["c"]
    inf = param["inf"]
    odwgt = param["odwgt"]

    # build model
    dual = Model("dual")

    # add variables
    u = dual.addVars([k for k in range(num_k)], [j for j in range(num_j)], vtype=GRB.CONTINUOUS, name="u")
    v = dual.addVars([k for k in range(num_k)], [j for j in range(num_j)], vtype=GRB.CONTINUOUS, name="v")
    w = dual.addVars([j for j in range(num_j)], lb=-1*GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")

    # set objective
    dual.setObjective(
        quicksum(   # \sum_j z_j(u,v)
            quicksum(   # z_j(u, v)
                quicksum(
                    (1 - u[k,j] + v[k,j]) * x[i,j]
                    for i in od[k][j]                                              
                ) + v[k,j] * maxwgt[j] * (1 - lr[j])
                for k in range(num_k)       
            ) - w[j]
            for j in range(num_j)
        ), GRB.MINIMIZE
    )

    # add constraints
    dual.addConstrs((
        quicksum(
            (u[k,j] - v[k,j])
            for k in range(num_k)
        ) * l * maxwgt[j] + w[j] <= 0
        for l in range(num_l)
        for j in range(num_j)
    ))

    dual.optimize()

    dual_u = np.zeros((num_k, num_j))
    dual_v = np.zeros((num_k, num_j))
    dual_w = np.zeros((num_j, 1))

    if dual.status == GRB.OPTIMAL:
        for j in range(num_j):
            print(w[j].x)
            dual_w[j] = w[j].x if w[j].x > 0 else 0
            for k in range(num_k):
                dual_u[k,j] = u[k,j].x if u[k,j].x > 0 else 0
                dual_v[k,j] = v[k,j].x if v[k,j].x > 0 else 0
    else:
        print("Not optimal: " + str(dual.status))
    return dual, dual_u, dual_v, dual_w


def iter_u(u, x, param):
    
    # read param
    num_j = param["j"]
    num_k = param["k"]
    od = param["od"]
    maxwgt = param["maxwgt"]
    alpha = param["alpha"]

    # iteration
    for k in range(num_k):
        for j in range(num_j):
            x_ij = sum([x[i, j] for i in od[k] ]) 
            q_j = x_ij // maxwgt[j] + 1
            loss = q_j * maxwgt[j] - x_ij
            u[k,j] -= alpha * loss
    return u


def iter_v(v, x, param):
    
    # read param
    num_j = param["j"]
    num_k = param["k"]
    od = param["od"]
    maxwgt = param["maxwgt"]
    lr = param["lr"]
    alpha = param["alpha"]

    # iteration
    for k in range(num_k):
        for j in range(num_j):
            x_ij = sum([x[i, j] for i in od[k] ]) 
            q_j = x_ij // maxwgt[j] + 1
            loss = xij - maxwgt[j] * (q_j - 1 + lr[j])
            v[k,j] -= alpha * loss
    return v


def iter_w(w, x, param):
    # read param
    num_j = param["j"]
    num_k = param["k"]
    maxwgt = param["maxwgt"]
    alpha = param["alpha"]
    for j in range(num_j):
        pass
    
    pass


# read data
param = {}
param["c"] = np.loadtxt("c.csv", delimiter="\t", dtype="int")
param["odwgt"] = np.loadtxt("od_wgt.csv", delimiter=",")
param["maxwgt"] = np.loadtxt("max_wgt.csv", delimiter=",")
param["od"] = [
    np.loadtxt("od1.csv", delimiter=",", dtype="int").tolist(),
    np.loadtxt("od2.csv", delimiter=",", dtype="int").tolist()
]

param["i"], param["j"] = param["c"].shape
param["k"] = 2  # od1 od2
param["l"] = 5  # max number of vehicles
param["inf"] = 1e8
param["lr"] = [0.5 for _ in range(param["j"])]

# read init solution
x = np.loadtxt("fessible_x.csv", delimiter=",")

# iter param
para["alpha"] = 0.0001