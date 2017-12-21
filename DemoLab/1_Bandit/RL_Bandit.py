import numpy as np
import Probability as p
from collections import defaultdict


max_epoch = 5

history = []
statistic = defaultdict(list)
reward = 0
time = 0
actions = ["1", "2", "3", "4", "5", "6"]

print("Game starts!")
while time < max_epoch:
    print("choose an action from 1 to 6")

    x = input()
    if x == "exit":
        print("Game Over")
        break

    # check history
    elif x == "history" or x == "h":
        print(history)

    elif x == "statistic" or x == "s":
        print(statistic)
        print("Mean reward: ")
        for k, v in statistic.items():
            print(k + " : " + str(np.mean(v)))

    elif x in actions:
        time += 1
        r = p.reward(int(x))
        print("Your reward is " + str(r))
        reward += (r - reward) / time
        history.append((time, x, r))
        statistic[x].append(r)

    else:
        print("illegal input")

# result
print("Show Result")
for k, v in statistic.items():
    print(k + " : " + str(np.mean(v)))
print(reward)
