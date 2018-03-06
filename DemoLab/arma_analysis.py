import numpy as np
import pandas as pd
import util
import statsmodels as sm
from matplotlib import pyplot as plt
import datetime as dt

pid = 7

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

df = util.add_special_date(df_day, ahead_effect=5, behind_effect=2)

df, aic_history = util.eliminate_outlier_df(df, 0.05, "sale_cnt", "daytime")

ts_sales = util.ggenerate_time_series(df, "sale_cnt_no_outlier", "daytime")

# acf and pacf
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
fig = sm.api.graphics.tsa.plot_acf(ts_sales, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.api.graphics.tsa.plot_pacf(ts_sales, ax=ax2)


