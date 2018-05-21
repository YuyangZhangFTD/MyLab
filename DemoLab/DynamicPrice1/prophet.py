"""
    prophet
"""
import numpy as np
import pandas as pd
import util
import statsmodels.api as sm
from matplotlib import pyplot as plt
import datetime as dt
from fbprophet import Prophet

pid = 4
split_train_end = "20171210"
split_test_start = "20171211"
arma_order = (3, 1)
ahead_period = 2

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

train = df_day[["daytime", "log_sale_cnt"]]
train.columns = ["ds", "y"]

m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
m.fit(train)
future = m.make_future_dataframe(periods=14)
forecast = m.predict(future)
m.plot(forecast)
plt.show()

