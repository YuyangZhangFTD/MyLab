import numpy as np


def reward(a):
    p = 0

    if a == 1:
        p = _action1()
    elif a == 2:
        p = _action2()
    elif a == 3:
        p = _action3()
    elif a == 4:
        p = _action4()
    elif a == 5:
        p = _action5()
    elif a == 6:
        p = _action6()

    if p < 0.4:
        return 500
    elif 0.4 <= p < 0.7:
        return 750
    elif 0.7 <= p < 0.8:
        return 1000
    elif 0.8 <= p < 0.85:
        return 2000
    elif 0.85 <= p < 0.9:
        return 2000
    elif 0.9 <= p < 0.95:
        return 4000
    else:
        return 500


#  uniform distribution U(0, 1)
def _action1():
    return np.random.uniform()


# normal distribution N(0.5, 1)
def _action2():
    p = np.random.standard_normal() + 0.5
    return p if (0 < p < 1) else 0


# normal distribution N(0.1, 2)
def _action3():
    p = np.random.standard_normal() * 2 + 0.1
    return p if (0 < p < 1) else 0


# beta distribution beta(0.5, 0.5)
def _action4():
    return np.random.beta(0.5, 0.5)


# exponential distribution
def _action5():
    p = np.random.exponential() / 10
    return p if p < 1 else 1


# normal distribution N(0.6, 0.5)
def _action6():
    p = np.random.standard_normal() / 2 + 0.6
    return p if (0 < p < 1) else 0
