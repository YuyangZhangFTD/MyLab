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


# train <---> fit
# test <---> prediction

# product id
pid = 5
# bootstrap number
B = 40
# test data length
gap = 4
# periods number
M = 2

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

data = df_day[["sale_cnt", "average_price"]].values

# train
sample_num = len(data)
week_num = int(sample_num / 7)
train_num = sample_num - gap * 7

train_data = data[:train_num]
test_data = data[train_num - M:]

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
bst_data = np.array([
    np.mean(data_split[i % 7][[np.random.randint(0, week_num - gap, B)]], axis=0)
    for i in range(train_num)
])

# first stage
first_var_num = M

xd_bst = np.zeros([train_num - M, first_var_num])
yd_bst = np.zeros([train_num - M, 1])
xd_bst[:, 0] = _log10(bst_data[M - 1:-1, 0])  # d_{t-1}
xd_bst[:, 1] = _log10(bst_data[M - 2:-2, 0])  # d_{t-2}
# xd_bst[:, 2] = _log10(bst_data[M - 3:-3, 0])  # d_{t-3}
yd_bst = _log10(bst_data[M:, 0])

result1 = sm.OLS(yd_bst, xd_bst).fit()
print(result1.summary())
wd = result1.params.reshape(first_var_num, 1)

# second stage
xd_train = np.zeros([train_num - M, first_var_num])
xd_train[:, 0] = _log10(train_data[M - 1:-1, 0])  # d_{t-1}
xd_train[:, 1] = _log10(train_data[M - 2:-2, 0])  # d_{t-2}
yd_fit = np.dot(xd_train, wd)

second_var_num = M + 1
xp_train = np.zeros([train_num - M, second_var_num])
yp_train = np.zeros([train_num - M, 1])

xp_train[:, 0] = _log10(train_data[M:, 1])  # p_{t}
xp_train[:, 1] = _log10(train_data[M - 1:-1, 1])  # p_{t-1}
xp_train[:, 2] = _log10(train_data[M - 2:-2, 1])  # p_{t-2}
yp_train = _log10(train_data[M:, 0].reshape(train_num - M, 1)) - yd_fit

result2 = sm.OLS(yp_train, xp_train).fit()
print(result2.summary())
wt = result2.params.reshape(second_var_num, 1)
yp_fit = np.dot(xp_train, wt)
y_fit = yp_fit + yd_fit

# test
test_num = len(test_data) - M
xd_test = np.zeros([test_num, first_var_num])
xd_test[:, 0] = _log10(test_data[M - 1:-1, 0])
xd_test[:, 1] = _log10(test_data[M - 2:-2, 0])
yd_prediction = np.dot(xd_test, wd)

yp_residual = _log10(test_data[M:, 0]).reshape(test_num, 1) - yd_prediction

xp_test = np.zeros([test_num, second_var_num])
xp_test[:, 0] = _log10(test_data[M:, 1])
xp_test[:, 1] = _log10(test_data[M - 1:-1, 1])
xp_test[:, 2] = _log10(test_data[M - 2:-2, 1])
yp_prediction = np.dot(xp_test, wt)

y_prediction = yd_prediction + yp_prediction

# count loss
y1 = np.power(10, y_prediction[:7]).reshape(7, 1)
y2 = test_data[M:M+7, 0].reshape(7, 1)
r_dbd = np.mean(np.abs(y1 - y2) / (y1 + y2 + np.ones((7, 1))))  # day by day prediction
print(r_dbd)

# ====================== PLOT ============================
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(range(M, train_num), _log10(train_data[M:, 0]), "b.-", label="train y")
ax1.plot(range(M, train_num), y_fit, "g.-", label="fit y")
ax1.plot(range(train_num, sample_num), _log10(test_data[M:, 0]), "r.-", label="test y")
ax1.plot(range(train_num, sample_num), y_prediction, "y.-", label="prediction y")
ax1.set_title("product_id " + str(pid) + " per day")
ax1.set_xlabel("time")
ax1.set_ylabel("log_sale_cnt")
ax1.legend()

ax2 = fig.add_subplot(212)
ax2.plot(range(M, train_num), yp_train, "b.-", label="train y residual")
ax2.plot(range(M, train_num), yp_fit, "g.-", label="fit y residual")
ax2.plot(range(train_num, sample_num), yp_residual, "r.-", label="test y residual")
ax2.plot(range(train_num, sample_num), yp_prediction, "y.-", label="prediction y residual")
ax2.set_title("residual fit with price data")
ax2.set_xlabel("time")
ax2.set_ylabel("log_sale_cnt")
ax2.legend()

plt.show()
