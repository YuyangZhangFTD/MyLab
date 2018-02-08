import pandas as pd
import datetime as dt
import numpy as np
from statsmodels import api as sm
from matplotlib import pyplot as plt
import util

# product id
pid = 4
# bootstrap number
B = 10
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
    for i in range(train_num)   # train_num or resample_num ?
])

# first stage
first_var_num = 3

xd = np.zeros([train_num - 7, first_var_num])
yd = np.zeros([train_num - 7, 1])
xd[:, 0] = np.log(demand_data[:-7, 0])           # d_{t-7}
xd[:, 1] = np.log(demand_data[6:-1, 0])          # d_{t-1}
xd[:, 2] = np.log(demand_data[7:, 1])            # p_{t}
yd = np.log(demand_data[7:, 0])

result1 = sm.OLS(yd, xd).fit()
print(result1.summary())
wd = result1.params.reshape(first_var_num, 1)
d_y = np.dot(xd, wd)

# second stage
xd2 = np.zeros([train_num - 7, 2])
xd2[:, 0] = np.log10(train_data[:-7, 0])                   # d_{t-7}
xd2[:, 1] = np.log10(train_data[6:-1, 0])                  # d_{t}
hat_d = np.dot(xd2, wd[:2])

second_var_num = 3
xt = np.zeros([train_num - 7, second_var_num])
yt = np.zeros([train_num - 7, 1])

xt[:, 0] = np.log10(train_data[7:, 1])                  # p_{t}
xt[:, 1] = np.log10(train_data[6:-1, 1])                 # p_{t-1}
xt[:, 2] = np.log10(train_data[5:-2, 1])                 # p_{t-2}
yt = np.max(np.log10(train_data[7:, 0].reshape(train_num-7, 1)), 0) - hat_d

result2 = sm.OLS(yt, xt).fit()
print(result2.summary())
wt = result2.params.reshape(second_var_num, 1)
hat_y = np.dot(xt, wt) + hat_d



"""
num_var = 2
x = np.zeros((len(data) - 7, num_var))
y = np.ones((len(data) - 7, 1))

x[:, 0] = data[6:-1, 1]  # p_{t-1}
x[:, 1] = data[5:-2, 1]  # p_{t-2}
# x[:, 2] = data[4:-3, 1]  # p_{t-3}
# x[:, 3] = data[3:-4, 1]  # p_{t-4}

y = data[7:, 0]  # d_{t}

data_num = len(y)
train_num = data_num - gap * 7
train_x = x[:train_num, :]
test_x = x[train_num:, :]
train_y = y[:train_num]
test_y = y[train_num:]

result = sm.OLS(train_y, train_x).fit()
print(result.summary())
w = result.params.reshape(num_var, 1)

y_hat = np.dot(train_x, w)
y_pre = np.dot(test_x, w)

y1 = np.power(10, y_pre[:7]).reshape(7, 1)
y2 = np.power(10, test_y[:7]).reshape(7, 1)

r = np.mean(np.abs(y1 - y2) / (y1 + y2 + np.ones((7, 1))))
print(r)

"""
# ====================== PLOT ============================
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(range(train_num), train_y, "r.-", label="train y")
# ax.plot(range(train_num, data_num), test_y, "b.-", label="test y")
# ax.plot(range(train_num), y_hat, "g.-", label="hat y")
# ax.plot(range(train_num, data_num), y_pre, "y.-", label="pre y")
ax.plot(range(train_num-7), np.log10(train_data[7:, 0]), "r.-", label="train y")
ax.plot(range(train_num-7), d_y, "y.-", label="demand y")
ax.plot(range(train_num-7), hat_y, "b.-", label="hat y")
ax.legend()
plt.show()

