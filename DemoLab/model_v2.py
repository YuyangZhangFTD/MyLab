"""
    two-stage method
    1st stage, fit bootstrapped d_{t-1} and d_{t-7} where p_t = \hat{p}_t^*
    2nd stage, fit p_t and p_{t-1}
"""
import pandas as pd
import datetime as dt
import numpy as np
from statsmodels import api as sm
from matplotlib import pyplot as plt
import util


def _log10(x):
    return np.maximum(np.log10(x), 0)


# product id
pid = 5
# bootstrap number
B = 20
#
gap = 2

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

data = df_day[["sale_cnt", "average_price"]].values

sample_num = len(data)
week_num = int(sample_num / 7)
train_num = sample_num - gap * 7

train_data = data[:train_num]
test_data = data[train_num:]

true_y = test_data[:, 1]

# Use all data to bootstrap \hat{p^*}
# data_split = [
#     data[range(0, sample_num, 7)]
#     for i in range(7)
# ]
# demand_data = np.array([
#     np.mean(data_split[i % 7][[np.random.randint(0, week_num, B)]], axis=0)
#     for i in range(sample_num)
# ])

# Use train data to bootstrap \hat{p^*}
data_split = [
    train_data[range(0, train_num, 7)]
    for i in range(7)
]
demand_data = np.array([
    np.mean(data_split[i % 7][[np.random.randint(0, week_num - gap, B)]], axis=0)
    for i in range(train_num)  # train_num or resample_num ?
])

# first stage
first_var_num = 3

xd = np.zeros([train_num - 2, first_var_num])
yd = np.zeros([train_num - 2, 1])
xd[:, 1] = _log10(demand_data[1:-1, 0])  # d_{t-1}
xd[:, 2] = _log10(demand_data[2:, 1])  # p_{t}
yd = _log10(demand_data[2:, 0])

result1 = sm.OLS(yd, xd).fit()
print(result1.summary())
wd = result1.params.reshape(first_var_num, 1)

# second stage
xd2 = np.zeros([train_num - 2, 2])
xd2[:, 0] = _log10(train_data[1:-1, 0])         # d_{t-1}
xd2[:, 1] = _log10(train_data[2:, 1])           # p_t
hat_d = np.dot(xd2, wd[:2])

second_var_num = 3
xt = np.zeros([train_num - 2, second_var_num])
yt = np.zeros([train_num - 2, 1])

xt[:, 0] = _log10(train_data[2:, 1])            # p_{t}
xt[:, 1] = _log10(train_data[1:-1, 1])          # p_{t-1}
xt[:, 2] = _log10(train_data[0:-2, 1])          # p_{t-2}
yt = _log10(train_data[2:, 0].reshape(train_num - 2, 1)) - hat_d

result2 = sm.OLS(yt, xt).fit()
print(result2.summary())
wt = result2.params.reshape(second_var_num, 1)
hat_y = np.dot(xt, wt)  # + hat_d

# ====================== PLOT ============================
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(range(train_num), train_y, "r.-", label="train y")
# ax.plot(range(train_num, data_num), test_y, "b.-", label="test y")
# ax.plot(range(train_num), y_hat, "g.-", label="hat y")
# ax.plot(range(train_num, data_num), y_pre, "y.-", label="pre y")
ax.plot(range(train_num - 2), _log10(train_data[2:, 0]), "r.-", label="train y")
ax.plot(range(train_num - 2), hat_d, "y.-", label="demand y")
ax.plot(range(train_num - 2), hat_y, "b.-", label="hat y")
ax.legend()
plt.show()
