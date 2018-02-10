"""
    over fitted
    residual linear regression
    d_t = f(p_t, d_{t-1}) + g(p_t, p_{t-1}, p_{t-2})
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
#
test_week_num = 2

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

data = df_day[["log_sale_cnt", "log_average_price"]].values

sample_num = len(data)
train_num = sample_num - test_week_num * 7

train_data = data[:train_num]
test_data = data[train_num-2:]

# first stage, fit demand
first_var_num = 2

xd = np.zeros([train_num - 2, first_var_num])
yd = np.zeros([train_num - 2, 1])
xd[:, 0] = train_data[1:-1, 0]      # d_{t-1}
xd[:, 1] = train_data[2:, 1]        # p_{t}
# xd[:, 2] = train_data[0:-2, 0]      # d_{t-2}
yd = train_data[2:, 0]

result1 = sm.OLS(yd, xd).fit()
print(result1.summary())
wd = result1.params.reshape(first_var_num, 1)

# second stage
hat_d = np.dot(xd, wd)

second_var_num = 3
xp = np.zeros([train_num - 2, second_var_num])
yp = np.zeros([train_num - 2, 1])

xp[:, 0] = train_data[2:, 1]        # p_{t}
xp[:, 1] = train_data[1:-1, 1]      # p_{t-1}
xp[:, 2] = train_data[0:-2, 1]      # p_{t-2}
yp = train_data[2:, 0].reshape(train_num - 2, 1) - hat_d

result2 = sm.OLS(yp, xp).fit()
print(result2.summary())
wp = result2.params.reshape(second_var_num, 1)
hat_y = np.dot(xp, wp) + hat_d

# predict
true_y = test_data[2:, 1]

test_num = len(test_data)

pre1_x = np.zeros([test_num - 2, first_var_num])
pre2_x = np.zeros([test_num - 2, second_var_num])
pre1_x[:, 0] = test_data[1:-1, 0]       # d_{t-1}
pre1_x[:, 1] = test_data[2:, 1]         # p_{t}
# pre1_x[:, 2] = test_data[0:-2, 0]       # d_{t-2}
pre2_x[:, 0] = test_data[2:, 1]         # p_{t}
pre2_x[:, 1] = test_data[1:-1, 1]       # p_{t-1}
pre2_x[:, 2] = test_data[0:-2, 1]       # p_{t-2}

pre_y = np.dot(pre1_x, wd) + np.dot(pre2_x, wp)

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
ax.plot(range(2, train_num), train_data[2:, 0], "r.-", label="train y")
ax.plot(range(2, train_num), hat_d, "y.-", label="demand y")
ax.plot(range(2, train_num), hat_y, "b.-", label="hat y")
ax.plot(range(train_num, sample_num), true_y, "r.-", label="test y")
ax.plot(range(train_num, sample_num), pre_y, "b.-", label="pre y")
ax.legend()
plt.show()
