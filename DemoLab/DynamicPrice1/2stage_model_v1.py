"""2-stage model
"""
import numpy as np
import pandas as pd
import util
from statsmodels import api as sm
from matplotlib import pyplot as plt
import datetime as dt

pid = 3
test_week_num = 10
is_log = False

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

df = util.add_special_date(df_day, ahead_effect=5, behind_effect=1)

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

ahead_period = 2

sample_num = len(data)
train_num = sample_num - test_week_num * 7
test_num = test_week_num * 7 + ahead_period

train_data = data[:train_num, :]
test_data = data[train_num - ahead_period:, :]

# train
var_num_1 = 3

x_1 = np.zeros((train_num - ahead_period, var_num_1))

# x_1[:, 0] = train_data[ahead_period - 1:-1, 0]
# x_1[:, 1] = train_data[ahead_period - 2:-2, 0]
#
x_1[:, 0] = train_data[ahead_period:, 1]  # p_{t}
x_1[:, 1] = train_data[ahead_period - 1:-1, 0]  # d_{t-1}
x_1[:, 2] = train_data[ahead_period - 1:-1, 1]  # p_{t-1}

y_1 = np.zeros((train_num - ahead_period, 1))
y_1[:, 0] = train_data[ahead_period:, 0]

result = sm.OLS(y_1, x_1).fit()
print(result.summary())

w_1 = result.params.reshape(var_num_1, 1)
hat_y_1 = np.dot(x_1, w_1)

var_num_2 = 6

x_2 = np.zeros((train_num - ahead_period, var_num_2))

x_2[:, 0] = np.power(train_data[ahead_period:, 1], 2)
x_2[:, 1] = np.power(train_data[ahead_period:, 1], 3)
x_2[:, 2] = np.power(train_data[ahead_period - 1:train_num - 1, 1], 2)
x_2[:, 3] = train_data[ahead_period:, 2]  # whether is a special day
x_2[:, 4] = train_data[ahead_period:, 3]  # whether is ahead a special day
x_2[:, 5] = train_data[ahead_period:, 4]  # whether is behind a special day

y_2 = np.zeros((train_num - ahead_period, 1))  # residual
y_2[:] = y_1 - hat_y_1

result = sm.OLS(y_2, x_2).fit()
print(result.summary())

w_2 = result.params.reshape(var_num_2, 1)
hat_y_2 = np.dot(x_2, w_2)

# test
test_y = np.zeros((sample_num - train_num, 1))
test_y[:, 0] = test_data[ahead_period:, 0]

test_x_1 = np.zeros((sample_num - train_num, var_num_1))
# test_x_1[:, 0] = test_data[ahead_period - 1:-1, 0]
# test_x_1[:, 1] = test_data[ahead_period - 2:-2, 0]
test_x_1[:, 0] = test_data[ahead_period:, 1]  # p_{t}
test_x_1[:, 1] = test_data[ahead_period - 1:-1, 0]  # d_{t-1}
test_x_1[:, 2] = test_data[ahead_period - 1:-1, 1]  # p_{t-1}

predict_y_1 = np.dot(test_x_1, w_1)

test_x_2 = np.zeros((sample_num - train_num, var_num_2))
test_x_2[:, 0] = np.power(test_data[ahead_period:, 1], 2)
test_x_2[:, 1] = np.power(test_data[ahead_period:, 1], 3)
test_x_2[:, 2] = np.power(test_data[ahead_period - 1:-1, 1], 2)
test_x_2[:, 3] = test_data[ahead_period:, 2]  # whether is a special day
test_x_2[:, 4] = test_data[ahead_period:, 3]  # whether is ahead a special day
test_x_2[:, 5] = test_data[ahead_period:, 4]  # whether is behind a special day

predict_y_2 = np.dot(test_x_2, w_2)

predict_y = predict_y_1 + predict_y_2

if is_log:
    evaluation_hat_y = np.power(10, np.array([float(x) for x in predict_y[:7]])).reshape(7, 1)
    evaluation_y = np.power(10, test_y[:7])

else:
    evaluation_hat_y = np.array([float(x) if x > 0 else 0 for x in predict_y[:7]]).reshape(7, 1)
    evaluation_y = test_y[:7]

evaluation = (np.abs(evaluation_y - evaluation_hat_y) / (evaluation_y + evaluation_hat_y + np.ones((7, 1))))
print(np.mean(evaluation))
print(evaluation_y)
print(evaluation_hat_y)
print(evaluation)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([i for i in range(ahead_period, train_num)], y_1, "r.-", label="train y")
ax.plot([i for i in range(ahead_period, train_num)], hat_y_1, "b.-", label="hat y")
ax.plot([i for i in range(train_num, sample_num)], test_y, "g.-", label="test y")
ax.plot([i for i in range(train_num, sample_num)], predict_y, "y.-", label="predict y")
ax.legend()
plt.show()
