import numpy as np
import pandas as pd
import util
from statsmodels import api as sm
from matplotlib import pyplot as plt
import datetime as dt

pid = 5
test_week_num = 6
alpha = 0.7

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

df = util.add_special_date(df_day, ahead_effect=7, behind_effect=5)

data = df[[
    "log_sale_cnt",
    "log_average_price",
    # "sale_cnt",
    # "average_price",
    "special_date",
    "ahead_special_date",
    "behind_special_date"
]].values

sample_num = len(data)
train_num = sample_num - test_week_num * 7

ahead_period = 1

train_data = data[:train_num]
test_data = data[train_num - ahead_period:]

x = np.zeros((train_num - ahead_period, 6))

x[:, 0] = train_data[ahead_period:, 1]  # p_{t}
x[:, 1] = train_data[ahead_period - 1:train_num - 1, 0]  # d_{t-1}
x[:, 2] = train_data[ahead_period - 1:train_num - 1, 1]  # p_{t-1}
x[:, 3] = train_data[ahead_period:, 2]  # whether is a special day
x[:, 4] = train_data[ahead_period:, 3]  # whether is ahead a special day
x[:, 5] = train_data[ahead_period:, 4]  # whether is behind a special day

y = np.zeros((train_num - ahead_period, 1))
y[:, 0] = train_data[ahead_period:, 0]

result1 = sm.OLS(y, x).fit()
print(" =================== result 1 =====================")
print(result1.summary())
w1 = result1.params.reshape(6, 1)
hat1_y = np.dot(x, w1).reshape((train_num - ahead_period, 1))
residual1 = y - hat1_y * alpha

result2 = sm.OLS(residual1, x).fit()
print(" =================== result 2 =====================")
print(result2.summary())
w2 = result2.params.reshape(6, 1)
hat2_y = np.dot(x, w2).reshape((train_num - ahead_period, 1))
residual2 = residual1 - hat2_y * alpha

# result3 = sm.OLS(residual2, x).fit()
# print(" =================== result 3 =====================")
# print(result3.summary())
# w3 = result3.params.reshape(6, 1)
# hat3_y = np.dot(x, w3).reshape((train_num - ahead_period, 1))
# residual3 = residual2 - hat3_y * alpha

# hat_y = alpha * np.dot(x, (w1 + w2 + w3))
hat_y = alpha * np.dot(x, w1) + np.dot(x, w2)

# predict
test_x = np.zeros((sample_num - train_num, 6))
test_x[:, 0] = test_data[ahead_period:, 1]  # p_{t}
test_x[:, 1] = test_data[ahead_period - 1:sample_num - train_num, 0]  # d_{t-1}
test_x[:, 2] = test_data[ahead_period - 1:sample_num - train_num, 1]  # p_{t-1}
test_x[:, 3] = test_data[ahead_period:, 2]  # whether is a special day
test_x[:, 4] = test_data[ahead_period:, 3]  # whether is ahead a special day
test_x[:, 5] = test_data[ahead_period:, 4]  # whether is behind a special day

test_y = np.zeros((sample_num - train_num, 1))
test_y[:, 0] = test_data[ahead_period:, 0]
# predict_y = alpha * np.dot(test_x, (w1 + w2 + w3))
predict_y = alpha * np.dot(test_x, w1) + np.dot(test_x, w2)


# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([i for i in range(ahead_period, train_num)], y, "r.-", label="train y")
ax.plot([i for i in range(ahead_period, train_num)], hat_y, "b.-", label="hat y")
ax.plot([i for i in range(train_num, sample_num)], test_y, "g.-", label="test y")
ax.plot([i for i in range(train_num, sample_num)], predict_y, "y.-", label="predict y")
ax.legend()
plt.show()
