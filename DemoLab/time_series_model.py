import pandas as pd
import datetime as dt
import numpy as np
from statsmodels import api as sm
from matplotlib import pyplot as plt
import util

pid = 5

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

data = df_day[["log_sale_cnt", "log_average_price"]].values
