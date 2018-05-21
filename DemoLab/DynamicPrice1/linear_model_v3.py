import numpy as np
import pandas as pd
import util
from statsmodels import api as sm
from matplotlib import pyplot as plt
import datetime as dt


pid = 7
test_week_num = 3
is_log = True

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

df = util.add_special_date(df_day, ahead_effect=5, behind_effect=2)

if is_log:
    data = df[[
        "log_sale_cnt",
        "log_average_price",
        "special_date",
        "ahead_special_date",
        "behind_special_date"
    ]].values
else:
    data = df[[
        "sale_cnt",
        "average_price",
        "special_date",
        "ahead_special_date",
        "behind_special_date"
    ]].values

sample_num = len(data)
train_num = sample_num - test_week_num * 7

ahead_period = 1

train_data = data[:train_num]
test_data = data[train_num - ahead_period:]

var_num = 5

x = np.zeros((train_num - ahead_period, var_num))

x[:, 0] = train_data[ahead_period:, 1]  # p_{t}
x[:, 1] = train_data[ahead_period - 1: -1, 0]  # d_{t-1}
x[:, 2] = train_data[ahead_period - 1: -1, 1]  # p_{t-1}
x[:, 3] = np.power(train_data[ahead_period:, 1], 2)
x[:, 4] = np.power(train_data[ahead_period:, 1], 3)

y = np.zeros((train_num - ahead_period, 1))
y = train_data[ahead_period:, 0]

result = sm.OLS(y, x).fit()
print(result.summary())

# predict
w = result.params.reshape(var_num, 1)
hat_y = np.dot(x, w)

test_x = np.zeros((sample_num - train_num, var_num))
test_x[:, 0] = test_data[ahead_period:, 1]  # p_{t}
test_x[:, 1] = test_data[ahead_period - 1: -1, 0]  # d_{t-1}
test_x[:, 2] = test_data[ahead_period - 1: -1, 1]  # p_{t-1}
test_x[:, 3] = np.power(test_data[ahead_period:, 1], 2)
test_x[:, 4] = np.power(test_data[ahead_period:, 1], 3)

test_y = np.zeros((sample_num - train_num, 1))
test_y[:, 0] = test_data[ahead_period:, 0]
predict_y = np.dot(test_x, w)

calculate_y = predict_y[:7]

if is_log:
    print(np.mean(
        np.abs(np.power(10, test_y[:7]) - np.power(10, calculate_y)) /
        (np.power(10, test_y[:7]) + np.power(10, np.max(calculate_y, 1)) + np.ones((7, 1)))
    ))
    print(np.power(10, test_y[:7]))
    print(np.power(10, calculate_y))
else:
    print(np.mean(
        np.abs(test_y[:7] - calculate_y) /
        (test_y[:7] + np.max(calculate_y, 0) + np.ones((7, 1)))
    ))
    print(test_y[:7])
    print(calculate_y)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([i for i in range(ahead_period, train_num)], y, "r.-", label="train y")
ax.plot([i for i in range(ahead_period, train_num)], hat_y, "b.-", label="hat y")
ax.plot([i for i in range(train_num, sample_num)], test_y, "g.-", label="test y")
ax.plot([i for i in range(train_num, sample_num)], predict_y, "y.-", label="predict y")
ax.legend()
plt.show()
