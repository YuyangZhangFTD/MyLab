"""
    linear regression
    just price
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
first_var_num = 4

xd = np.zeros([train_num - 2, first_var_num])
yd = np.zeros([train_num - 2, 1])
xd[:, 0] = train_data[1:-1, 0]      # d_{t-1}
xd[:, 1] = train_data[2:, 1]        # p_{t}
# xd[:, 2] = train_data[0:-2, 0]      # d_{t-2}
yd = train_data[2:, 0]

result1 = sm.OLS(yd, xd).fit()
print(result1.summary())
wd = result1.params.reshape(first_var_num, 1)


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
"""
