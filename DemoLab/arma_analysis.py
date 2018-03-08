import numpy as np
import pandas as pd
import util
import statsmodels as sm
from matplotlib import pyplot as plt
import datetime as dt

pid = 7

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

df, aic_history = util.eliminate_outlier_df(df, 0.05, "sale_cnt", "daytime")

ts_sales = util.ggenerate_time_series(df, "sale_cnt_no_outlier", "daytime")

ts_sales_train = ts_sales[:split_train_end]
ts_sales_test = ts_sales[split_test_start:]

train_num = len(ts_sales_train)
test_num = len(ts_sales_test)

arma_model = sm.tsa.ARMA(ts_sales_train, arma_order)
ts_result = arma_model.fit(disp=False)
print(ts_result.summary())

arma_hat = arma_model.predict(ts_result.params, start=start_date, end=split_train_end)
arma_predict = arma_model.predict(ts_result.params, start=split_test_start, end=end_date)[:-1]

# acf and pacf
fig = plt.figure()
ax1 = fig.add_subplot(211)
fig = sm.api.graphics.tsa.plot_acf(ts_sales, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.api.graphics.tsa.plot_pacf(ts_sales, ax=ax2)


