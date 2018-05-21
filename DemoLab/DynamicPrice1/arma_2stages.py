"""
    ARMA+linear model
    d_t = ARMA(sales) + f(sales, price, special day)
"""
import numpy as np
import pandas as pd
import util
import statsmodels.api as sm
from matplotlib import pyplot as plt
import datetime as dt

pid = 4
split_train_end = "20171203"
split_test_start = "20171204"
arma_order = (3, 1)
ahead_period = 2

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

df = util.add_special_date(df_day, ahead_effect=5, behind_effect=2)

df, aic_history = util.eliminate_outlier_df(df, 0.05, "sale_cnt", "daytime", "sale_cnt_no_outlier")

ts_sales = util.generate_time_series(df, "sale_cnt_no_outlier", "daytime")

ts_sales_train = ts_sales[:split_train_end]
ts_sales_test = ts_sales[split_test_start:]

train_num = len(ts_sales_train)
test_num = len(ts_sales_test)

arma_model = sm.tsa.ARMA(ts_sales_train, arma_order)
ts_result = arma_model.fit(disp=False)
print(ts_result.summary())

arma_hat = arma_model.predict(ts_result.params, start=start_date, end=split_train_end)
arma_predict = arma_model.predict(ts_result.params, start=split_test_start, end=end_date)[:-1]

data = df[[
    # "log_sale_cnt",
    # "log_average_price",
    "sale_cnt",
    "average_price",
    "special_date",
    "ahead_special_date",
    "behind_special_date"
]].values
train_data = data[:train_num]
test_data = data[train_num - ahead_period:]

x = np.zeros((train_num - ahead_period, 3))
x[:, 0] = train_data[ahead_period:, 1]  # p_{t}
x[:, 1] = train_data[ahead_period - 1: -1, 0]  # d_{t-1}
x[:, 2] = train_data[ahead_period - 1: -1, 1]  # p_{t-1}
# x[:, 3] = train_data[ahead_period:, 2]  # whether is a special day
# x[:, 4] = train_data[ahead_period:, 3]  # whether is ahead a special day
# x[:, 5] = train_data[ahead_period:, 4]  # whether is behind a special day

y = np.ones((train_num - ahead_period, 1))
y[:, 0] = train_data[ahead_period:, 0] - arma_hat.reshape(train_num, 1)[ahead_period:, 0]

ols_model = sm.OLS(y, x)
ols_result = ols_model.fit()
print(ols_result.summary())

ols_w = ols_result.params
ols_hat = np.dot(x, ols_w)

x_test = np.zeros((test_num, 3))
x_test[:, 0] = test_data[ahead_period:, 1]  # p_{t}
x_test[:, 1] = test_data[ahead_period - 1: -1, 0]  # d_{t-1}
x_test[:, 2] = test_data[ahead_period - 1: -1, 1]  # p_{t-1}
# x_test[:, 3] = test_data[ahead_period:, 2]  # whether is a special day
# x_test[:, 4] = test_data[ahead_period:, 3]  # whether is ahead a special day
# x_test[:, 5] = test_data[ahead_period:, 4]  # whether is behind a special day

y_test = np.ones((test_num, 1))
y_test[:, 0] = test_data[ahead_period:, 0]  # d_t

y_ols_predict = np.ones((test_num, 1))
y_ols_predict[:, 0] = np.dot(x_test, ols_w)

y_predict = y_ols_predict + arma_predict.reshape((test_num, 1))

print(np.abs(y_predict[0, 0] - y_test[0, 0]) / (y_predict[0, 0] + y_test[0, 0] + 1))

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([i for i in range(ahead_period, train_num)], y, "r-.", label="train y")
ax1.plot(
    [i for i in range(ahead_period, train_num)],
    arma_hat.reshape(train_num, 1)[ahead_period:, 0],
    "b-.",
    label="amra y"
)
ax1.plot(
    [i for i in range(ahead_period, train_num)],
    ols_hat + arma_hat.reshape(train_num, 1)[ahead_period:, 0],
    "g-.",
    label="hat y"
)
ax1.plot(
    [i for i in range(train_num, train_num + test_num)],
    y_test, "r.-", label="test y"
)
ax1.plot(
    [i for i in range(train_num, train_num + test_num)],
    arma_predict.reshape((test_num, 1)),
    "b.-", label="arma predict"
)
ax1.plot(
    [i for i in range(train_num, train_num + test_num)],
    y_predict, "g.-", label="predict y"
)
ax1.legend()
plt.show()
