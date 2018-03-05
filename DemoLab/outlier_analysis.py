"""
    outliers analysis
"""
import pandas as pd
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
import util


def _aic(param_ts, param_n, param_alpha):
    return param_alpha * param_n + np.log(param_ts.std())


pid = 4
alpha = 0.02
# alpha_log = 0.1

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

# origin time series
ts_sales = util.generate_time_series(df_day, "sale_cnt", "daytime")
ts_sales_zero_mean = ts_sales - ts_sales.mean()

# ts_log_sales = util.generate_time_series(df_day, "log_sale_cnt", "daytime")
# ts_log_sales_zero_mean = ts_log_sales - ts_log_sales.mean()

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1.plot(ts_sales, "r.-", label="sale_cnt")
ax1.plot(ts_sales_zero_mean, "g.-", label="sale_cnt_zero_mean")

# ax2 = fig.add_subplot(312)
# ax2.plot(ts_log_sales, "r.-", label="log_sale_cnt")
# ax2.plot(ts_log_sales_zero_mean, "g.-", label="log_sale_cnt_zero_mean")

old_aic = _aic(ts_sales, 0, alpha)
n_outlier = 0
aic_history = []
while True:
    outlier_index = ts_sales_zero_mean.abs().argmax()
    print("="*50)
    print("outlier index  " + str(outlier_index))
    print("outlier in index list ?  " + str(outlier_index in ts_sales.index))
    if outlier_index in ts_sales.index:
        outlier_value = ts_sales_zero_mean[outlier_index]
        print("outlier value  " + str(outlier_value))
        print("std before eliminating outlier  " + str(ts_sales_zero_mean.std()))
        ts_sales_zero_mean[outlier_index] = None
        n_outlier += 1
        print("std after eliminating outlier  " + str(ts_sales_zero_mean.std()))
        print("new aic  " + str(_aic(ts_sales_zero_mean, n_outlier, alpha)))
        print("old aic  " + str(old_aic))
        if _aic(ts_sales_zero_mean, n_outlier, alpha) > old_aic:
            break
        else:
            ts_sales[outlier_index] = None
            old_aic = _aic(ts_sales_zero_mean, n_outlier, alpha)
            aic_history.append((outlier_index, outlier_value))
    else:
        break

# origin_aic_log = _aic(ts_sales, 0, alpha_log)
# n_outlier_log = 0
# aic_history_log = []
# while True:
#     outlier_index = ts_log_sales_zero_mean.abs().argmax()
#     if outlier_index in ts_sales.index:
#         outlier_value = ts_log_sales_zero_mean[outlier_index]
#         ts_log_sales_zero_mean[outlier_index] = None
#         n_outlier_log += 1
#         if _aic(ts_log_sales_zero_mean, n_outlier_log, alpha_log) > origin_aic_log:
#             break
#         else:
#             ts_log_sales[outlier_index] = None
#             aic_history_log.append((outlier_index, outlier_value))
#     else:
#         break

ts_sales_no_outlier = ts_sales.fillna(ts_sales.mean())
# ts_log_sales_no_outlier = ts_log_sales.fillna(ts_log_sales.mean())

ax1.plot(ts_sales_no_outlier, "c.-", label="sale_cnt_no_outlier")
ax1.plot(ts_sales_no_outlier - ts_sales_no_outlier.mean(), "y.-", label="sale_cnt_zero_mean_no_outlier")
ax1.legend()

# ax2.plot(ts_log_sales, "c.-", label="log_sale_cnt")
# ax2.plot(ts_log_sales_zero_mean, "y.-", label="log_sale_cnt_zero_mean")
# ax2.legend()

ax3 = fig.add_subplot(325)
ax3.plot([i for i in range(n_outlier-1)], [x[1] for x in aic_history])\

# ax4 = fig.add_subplot(326)
# ax4.plot([i for i in range(n_outlier_log-1)], [x[1] for x in aic_history_log])

plt.show()
