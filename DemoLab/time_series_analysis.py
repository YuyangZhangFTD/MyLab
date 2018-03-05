import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
from matplotlib import pyplot as plt
import util

pid = 16
split_date = "20171201"

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

# origin time series
ts_sales = util.generate_time_series(df_day, "sale_cnt", "daytime")
ts_log_sales = util.generate_time_series(df_day, "log_sale_cnt", "daytime")

# train set and test set
train_ts_sales = ts_sales[:split_date]
test_ts_sales = ts_sales[split_date:]
train_ts_log_sales = ts_log_sales[:split_date]
test_ts_sales = ts_log_sales[split_date:]

# pacf
fig = plt.figure()
ax1 = fig.add_subplot(321)
fig = sm.graphics.tsa.plot_acf(ts_sales, lags=80, ax=ax1)
ax2 = fig.add_subplot(322)
fig = sm.graphics.tsa.plot_pacf(ts_sales, lags=80, ax=ax2)

ax3 = fig.add_subplot(323)
fig = sm.graphics.tsa.plot_acf(ts_log_sales, lags=40, ax=ax3)
ax4 = fig.add_subplot(324)
fig = sm.graphics.tsa.plot_pacf(ts_log_sales, lags=40, ax=ax4)

period_num = len(ts_sales)

# fig = plt.figure()
ax5 = fig.add_subplot(325)
ax5.plot([i for i in range(period_num)], ts_sales.values, "r.-", label="ts_sales")
ax5.plot([i for i in range(1, period_num)], ts_sales.diff(1).values[1:], "b.-", label="ts_sales_diff1")
ax5.plot([i for i in range(2, period_num)], ts_sales.diff(2).values[2:], "g.-", label="ts_sales_diff2")
ax5.plot([i for i in range(3, period_num)], ts_sales.diff(3).values[3:], "y.-", label="ts_sales_diff3")
ax5.legend()
ax6 = fig.add_subplot(326)
ax6.plot([i for i in range(period_num)], ts_log_sales.values, "r.-", label="ts_log_sales")
ax6.plot([i for i in range(1, period_num)], ts_log_sales.diff(1).values[1:], "b.-", label="ts_log_sales_diff1")
ax6.plot([i for i in range(2, period_num)], ts_log_sales.diff(2).values[2:], "g.-", label="ts_log_sales_diff2")
ax6.plot([i for i in range(3, period_num)], ts_log_sales.diff(3).values[3:], "y.-", label="ts_log_sales_diff3")
ax6.legend()
plt.show()


