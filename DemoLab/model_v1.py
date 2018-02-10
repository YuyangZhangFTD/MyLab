"""
    sample linear regression
    d_{t-1}, p_{t-1}, p_t are considered
"""
import pandas as pd
import datetime as dt
import numpy as np
from statsmodels import api as sm
from matplotlib import pyplot as plt
import util

pid = 191

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

# data = df_day[["sale_cnt", "average_price"]].values  # (sale_cnt, average_price)
data = df_day[["log_sale_cnt", "log_average_price"]].values
# data = df_week[["log_sale_cnt", "log_average_price"]].values

num_var = 2
x = np.zeros((len(data) - 3, num_var))
y = np.ones((len(data) - 3, 1))

x[:, 0] = data[2:-1, 0]     # d_{t-1}
x[:, 1] = data[3:, 1]  # p_{t}
# x[:, 2] = data[2:-1, 1]  # p_{t-1}
# x[:, 3] = data[:-2, 0]    # d_{t-2}
# x[:, 4] = data[:-2, 1]    # p_{t-2}

y = data[3:, 0]  # d_{t}

data_num = len(y)
train_num = int(data_num * 0.8)
train_x = x[:train_num, :]
test_x = x[train_num:, :]
train_y = y[:train_num]
test_y = y[train_num:]

result = sm.OLS(train_y, train_x).fit()
print(result.summary())
w = result.params.reshape(num_var, 1)

y_hat = np.dot(train_x, w)
y_pre = np.dot(test_x, w)


import model_v1_test as m
max_y = np.max(train_y)
min_y = 0
period = 7
func = m._generate_model(w)
yy = m._predict_util(func, test_x, period, max_y, min_y)


y1 = np.power(10, y_pre[:7]).reshape(7, 1)
y2 = np.power(10, test_y[:7]).reshape(7, 1)
yy1 = np.power(10, yy[:7]).reshape(7, 1)

r = np.mean(np.abs(y1 - y2) / (y1 + y2 + np.ones((7, 1))))
print(r)
r = np.mean(np.abs(yy1 - y2) / (yy1 + y2 + np.ones((7, 1))))
print(r)

# ====================== PLOT ============================
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(train_num), train_y, "r.-", label="train y")
ax.plot(range(train_num, data_num), test_y, "b.-", label="test y")
ax.plot(range(train_num), y_hat, "g.-", label="hat y")
ax.plot(range(train_num, data_num), y_pre, "y.-", label="pre y")
ax.plot(range(train_num, train_num+period), yy, "c.-", label="yy")
ax.legend()
plt.show()
