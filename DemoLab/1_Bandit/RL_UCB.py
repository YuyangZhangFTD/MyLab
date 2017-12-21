import numpy as np
import math as m
import Probability as p
from collections import defaultdict

max_epoch = 50
alpha = 2
epsilon = 0.1
init_value = 400


def calculate(tpl_, time_, c):
    value = tpl_[2] + (m.log(time_) / tpl_[1]) ** 0.5 * c
    return value, tpl_


history = []
statistic = defaultdict(list)
reward = 0
time = 0
mean_reward = []

actions = ["1", "2", "3", "4", "5", "6"]

# init
for a in actions:
    statistic[a].append(init_value)
    mean_reward.append((a, 1, init_value))

while time < max_epoch:

    time += 1

    # ====================================================================
    # choose action
    if np.random.uniform(0, 1) < epsilon:
        l = []
        for tpl in mean_reward:
            l.append(calculate(tpl, time, alpha))
        print(l)
        np.random.shuffle(l)
        mean_reward.sort(key=lambda x: x[0])
        a = mean_reward[-1][0]
    else:
        a = np.random.choice(actions)
    # ====================================================================

    # update
    r = p.reward(int(a))
    reward += (r - reward) / time
    history.append((time, a, r))
    statistic[a].append(r)

    for i in range(len(actions)):
        a_, t_, r_ = mean_reward[i]
        if a_ == a:
            mean_reward[i] = (a_, t_ + 1, r_ + (r - r_) / t_)
            break

# result
print(mean_reward)
print(reward)
