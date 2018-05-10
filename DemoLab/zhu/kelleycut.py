import numpy as np
from matplotlib import pyplot as plt

# hyper-parameters
epsilon = 0.0001
iter_num = 100


# (a) kelley'c cut framework
'''
def f(x):
    fx = None
    return fx


def df(x):
    dfx = None
    return dfx


def cal_a(x):
    return df(x)


def cal_b(a, x, y):
    b = y - a * x
    return b


def cal_x(a, b, y=0):
    x = (y - b) / a
    return x


def cal_intersection(hp1, hp2):
    a1, b1 = hp1
    a2, b2 = hp2
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return x, y


UB = []
LB = []
x_history = None
hyperplane_history = None
x_bound = {"lb": None, "ub": None}
for iter_i in range(1, iter_num):

    # stop rule
    if np.abs(UB[-1] - LB[-1]) <= epsilon:
        break

    # get latest x
    x = x_history[-1]

    # calculate hyperplane ax+y=b
    y = f(x)
    a = cal_a(x)
    b = cal_b(a, x, y)

    # calculate new x
    # one of these cases will hold:
    #   a. new x will be found where y=0
    #   b. new x will be found at the intersection of two hyperplanes
    #   c. new x will be found at the bound of the feasible region of x
    compare_xy = []  # (x, lb, ub, is_hyperplane)
    for (a2, b2) in hyperplane_history:
        intersection_x, intersection_y = cal_intersection((a, b), (a2, b2))
        # check feasible region of x
        if intersection_x < x_bound["lb"]:
            compare_xy.append((x_bound["lb"], 0, f(x_bound["lb"]), False))
        elif intersection_x > x_bound["ub"]:
            compare_xy.append((x_bound["ub"], 0, f(x_bound["ub"]), False))
        else:
            compare_xy.append((intersection_x, intersection_y, f(intersection_x), True))
    else:  # if constraint set is empty or hyperplane_history is empty
        x_0 = cal_x(a, b, 0)
        compare_xy.append((x_0, 0, f(x_0), True))

    # choose next x by comparing upper bound
    compare_xy.sort(key=lambda tpl: tpl[-2])

    # record hyperplane history
    # choose the maximal y of the candidate points
    if compare_xy[0][-1]:
        hyperplane_history.append((a, b))
    # record new x
    x_history.append(compare_xy[0][0])
    # new upper bound equal to f(x^k)
    # new lower bound equal to y^k of the latest intersection
    UB.append(compare_xy[0][2])
    LB.append(compare_xy[0][1])

    # report
    print("=" * 50)
    print("iteration k : " + str(iter_i))
    print("x^k : " + str(x_history[-1]))
    print("f_k(x^k) : " + str(LB[-1]))
    print("f(x^k) : " + str(UB[-1]))
'''


# (b) solve the following question
# min   f(x) = (x-2)^2+1
# s.t.  -1 <= x <= 4
# hyperplane    ax+b=y
def f(x):
    fx = (x - 2) ** 2 + 1
    return fx


def df(x):
    dfx = 2 * x - 4
    return dfx


def cal_a(x):
    return df(x)


def cal_b(a, x, y):
    b = y - a * x
    return b


def cal_x(a, b, y=0):
    x = (y - b) / a
    return x


def cal_intersection(hp1, hp2):
    a1, b1 = hp1
    a2, b2 = hp2
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return x, y


x_history = [-1]
hyperplane_history = []
x_bound = {"lb": -1, "ub": 4}
UB = [f(x_history[0])]
LB = [0]
for iter_i in range(1, iter_num):

    # stop rule
    if np.abs(UB[-1] - LB[-1]) <= epsilon:
        break

    # get latest x
    x = x_history[-1]

    # calculate hyperplane ax+y=b
    y = f(x)
    a = cal_a(x)
    b = cal_b(a, x, y)

    # calculate new x
    # one of these cases will hold:
    #   a. new x will be found where y=0
    #   b. new x will be found at the intersection of two hyperplanes
    #   c. new x will be found at the bound of the feasible region of x
    compare_xy = []  # (x, lb, ub, is_hyperplane)
    for (a2, b2) in hyperplane_history:
        intersection_x, intersection_y = cal_intersection((a, b), (a2, b2))
        # check feasible region of x
        if intersection_x < x_bound["lb"]:
            compare_xy.append((x_bound["lb"], 0, f(x_bound["lb"]), False))
        elif intersection_x > x_bound["ub"]:
            compare_xy.append((x_bound["ub"], 0, f(x_bound["ub"]), False))
        else:
            compare_xy.append((intersection_x, intersection_y, f(intersection_x), True))
    else:  # if constraint set is empty or hyperplane_history is empty
        x_0 = cal_x(a, b, 0)
        compare_xy.append((x_0, 0, f(x_0), True))

    # choose next x by comparing upper bound
    compare_xy.sort(key=lambda tpl: tpl[-2])

    # record hyperplane history
    # choose the maximal y of the candidate points
    if compare_xy[0][-1]:
        hyperplane_history.append((a, b))
    # record new x
    x_history.append(compare_xy[0][0])
    # new upper bound equal to f(x^k)
    # new lower bound equal to y^k of the latest intersection
    UB.append(compare_xy[0][2])
    LB.append(compare_xy[0][1])

    # report
    print("=" * 50)
    print("iteration k : " + str(iter_i))
    print("x^k : " + str(x_history[-1]))
    print("f_k(x^k) : " + str(LB[-1]))
    print("f(x^k) : " + str(UB[-1]))

# result
print("Optimal solution: ")
print(x_history[-1])
print("optimal value: ")
print(f(x_history[-1]))

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plot_x = np.linspace(-1, 4, 100)
plot_y = (plot_x - 2) ** 2 + 1   # f(x)
ax.plot(plot_x, plot_y)
ax.scatter(x_history, [f(x) for x in x_history], s=100, c="r", marker="*")
plt.show()
