import numpy as np
from gurobipy import *


param = {}

param["c"] = np.loadtxt("c.csv", delimiter="\t")
param["od_wgt"] = np.loadtxt("od_wgt.csv", delimiter=",")
param["max_wgt"] = np.loadtxt("max_wgt.csv", delimiter=",")
param["od"] = [
    np.loadtxt("od1.csv", delimiter=",", dtype="int"),
    np.loadtxt("od2.csv", delimiter=",", dtype="int")
]
param["loading_rate"] = [0.5 for _ in range(param["c"].shape[1])]

def SP(bar_q, Q, E, iter_i, parameter):
    c = parameter["c"]
    od_wgt = parameter["od_wgt"]
    max_wgt = parameter["max_wgt"]
    od = parameter["od"]
    num_i, num_j = parameter["c"].shape
    loading_rate = parameter["loading_rate"]
    stop = False    

    # dual problem of slave problem
    print("=" * 20 + " slave problem " + "=" * 20)
    sp = Model("dual_slave")

    # for getting unboundedRay
    sp.Params.InfUnbdInfo = 1

    # build model
    x = sp.addVars([i for i in range(num_i)], [j for j in range(num_j)], vtype=GRB.CONTINUOUS, name="x")
    obj = sp.setObjective(
        quicksum(
            x[i,j] for i in range(num_i) for j in range(num_j)
        ), GRB.MAXIMIZE
    )
    u = sp.addConstrs((
        x[i, j] <= c[i,j] * od_wgt[i]
        for i in range(num_i)
        for j in range(num_j)
    ))
    v = sp.addConstrs((
        quicksum(
            x[i,j] for j in range(num_j)
        ) <= od_wgt[i] 
        for i in range(num_i)
    ))
    y = sp.addConstrs((
        quicksum(
            x[i,j] for i in od[k][j,:]
        ) <= max_wgt[j] * bar_q[j]
        for k in [0, 1]
        for j in range(num_j)
    ))
    z = sp.addConstrs((
        quicksum(
            x[i,j] for i in od[k][j,:]
        ) >= max_wgt[j] * (bar_q[j] - 1 + loading_rate[j])
        for k in [0, 1]
        for j in range(num_j)
    ))

    sp.optimize()

    item = dict()
    if sp.status == GRB.INFEASIBLE:
        item["u"] = {
            (i,j): u[i, j].farkasdual
            for j in range(num_j)
            for i in range(num_i)
        }
        item["v"] = {
            i: v[i].farkasdual
            for i in range(num_i)
        }
        item["y"] = {
            (k,j): y[k, j].farkasdual
            for k in [0, 1]
            for j in range(num_j)
        }
        item["z"] = {
            (k,j): z[k, j].farkasdual
            for k in [0, 1]
            for j in range(num_j)
        }
        Q[iter_i] = item
        print("Add feasible cut")
    elif sp.status == GRB.OPTIMAL:
        item["u"] = {
            (i,j): u[i, j].pi
            for j in range(num_j)
            for i in range(num_i)
        }
        item["v"] = {
            i: v[i].pi
            for i in range(num_i)
        }
        item["y"] = {
            (k,j): y[k, j].pi
            for k in [0, 1]
            for j in range(num_j)
        }
        item["z"] = {
            (k,j): z[k, j].pi
            for k in [0, 1]
            for j in range(num_j)
        }
        E[iter_i] = item
        print("Add optimal cut")
    else:
        stop = True
        print("wrong slave sub-problem status  " + str(sp.status))

    return sp, stop


def MP(Q, E, iter_i, parameter):
    c = parameter["c"]
    od_wgt = parameter["od_wgt"]
    max_wgt = parameter["max_wgt"]
    od = parameter["od"]
    num_i, num_j = parameter["c"].shape
    loading_rate = parameter["loading_rate"]
    stop = False

    # master problem
    print("=" * 20 + " master problem " + "=" * 20)
    mp = Model()

    q = mp.addVars([j for j in range(num_j)], vtype=GRB.INTEGER, name="q")
    theta = mp.addVar(name="theta")

    mp.setObjective(theta, GRB.MAXIMIZE)

    # prior knowledge
    mp.addConstr(quicksum(q[j] for j in range(num_j)) >= 5)
    mp.addConstrs((
        q[j] <= quicksum(
            od_wgt[i]
            for i in od[k][j,:]
        ) / max_wgt[j] + 1
        for k in [0, 1]
        for j in range(num_j)
    ))
    
    # cuts
    for item in Q.values():
        mp.addConstr(
            quicksum(
                c[i,j] * od_wgt[i] * item["u"][i,j]
                for i in range(num_i)
                for j in range(num_j)
            ) + quicksum(
                od_wgt[i] * item["v"][i]
                for i in range(num_i)
            ) + quicksum(
                q[j] * max_wgt[j] * item["y"][k, j]
                for k in [0, 1]
                for j in range(num_j)
            ) + quicksum(
                (1 - q[j] + loading_rate[j]) * max_wgt[j] * item["z"][k, j]
                for k in [0, 1]
                for j in range(num_j)
            ) <= 0
        )
    for item in E.values():
        mp.addConstr(
            quicksum(
                c[i,j] * od_wgt[i] * item["u"][i,j]
                for i in range(num_i)
                for j in range(num_j)
            ) + quicksum(
                od_wgt[i] * item["v"][i]
                for i in range(num_i)
            ) + quicksum(
                q[j] * max_wgt[j] * item["y"][k, j]
                for k in [0, 1]
                for j in range(num_j)
            ) + quicksum(
                (1 - q[j] - loading_rate[j]) * max_wgt[j] * item["z"][k, j]
                for k in [0, 1]
                for j in range(num_j)
            ) >= theta
        )

    mp.optimize()

    bar_q = dict()
    if mp.status == GRB.OPTIMAL:
        for j in range(num_j):
            bar_q[j] = q[j].x
    else:
        print("There is no feasible solution in primal problem")
        stop = True

    return mp, bar_q, stop


E_set = dict()
Q_set = dict()

LB = -1 * 1e10
UB = 1e10

warm_start = True
feasible_cnt = 0
optimal_cnt = 0

bar_q = np.loadtxt("feasible_q.csv")

for iter_i in range(1000):

    if np.abs(UB - LB) < 0.01:
        print("Optimal")
        break

    print("=" * 100)
    print("iteration at " + str(iter_i))

    sp, stop = SP(Q_set, E_set, bar_q, iter_i, param)

    print("Q  " + str(len(Q_set.keys())))
    print("E  " + str(len(E_set.keys())))

    if stop:
        print("Wong slave problem")
        break

    item = E_set.get(iter_i, False)
    if item:
        c = param["c"]
        od_wgt = param["od_wgt"]
        max_wgt = param["max_wgt"]
        od = param["od"]
        num_i, num_j = param["c"].shape
        loading_rate = param["loading_rate"]

        dual_optimal = sum([
            c[i,j] * od_wgt[i] * item["u"][i,j]
            for i in range(num_i)
            for j in range(num_j)
        ]) + sum([
            od_wgt[i] * item["v"][i]
            for i in range(num_i)
        ]) + sum([
            bar_q[j] * max_wgt[j] * item["y"][k, j]
            for k in [0, 1]
            for j in range(num_j)
        ]) + sum([
            (1 - bar_q[j] - loading_rate[j]) * max_wgt[j] * item["z"][k, j]
            for k in [0, 1]
            for j in range(num_j)
        ])

        print("slave dual objective value")
        print(dual_optimal)

        UB = min(UB, dual_optimal)
    
    master, bar_q, stop = MP(Q_set, E_set, iter_i, param)

    print("master objective value")
    print(master.objVal)

    if stop:
        print("wrong master problem")
        break

    LB = master.objVal

    print("UB " + str(UB))
    print("LB " + str(LB))