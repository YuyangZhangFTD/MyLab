"""

"""
import numpy as np
import pandas as pd
import util
from statsmodels import api as sm
from matplotlib import pyplot as plt
import datetime as dt


pid = 191
test_week_num = 3
is_log = False

start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
df_date = util.generate_time_index_df(start_date, end_date)

f = pd.read_csv("input/" + str(pid) + ".csv")
df_day = util.handle_original_df_day_sn(f, df_date)
df_week = util.handle_original_df_week_sn(df_day)

df = util.add_special_date(df_day, ahead_effect=5, behind_effect=2)

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

sample_num = len(data)
train_num = sample_num - test_week_num * 7

ahead_period = 1

train_data = data[:train_num]
test_data = data[train_num - ahead_period:]


