import numpy as np
import math as m
import Probability as p
from collections import defaultdict

max_epoch = 50
alpha = 0.1
init_value = 400
epsilon = 0.1


def softmax(v):
    return np.exp(v) / sum(np.exp(v))


vec = np.ones((6, 1)) * init_value

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
        pi = softmax(vec)
        a = vec.argmax()
    else:
        pi = np.ones((6, 1)) / 6
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
            vec[int(a_) - 1] += alpha * (r - reward) * (1 - pi[int(a_) - 1])
        else:
            vec[int(a_) - 1] -= alpha * (r - reward) * pi[int(a_) - 1]

# result
print(mean_reward)
print(reward)
