import util
import pandas as pd
import datetime as dt
import numpy as np
from collections import defaultdict

# read test id list
pid_df = pd.read_csv("input/id_list.csv")
id_list = pid_df["id"].values.tolist()  # 135 ids

record_3weeks = dict()
record_10weeks = dict()

for pid in id_list:
    start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
    df_date = util.generate_time_index_df(start_date, end_date)

    f = pd.read_csv("input/" + str(pid) + ".csv")
    df_day = util.handle_original_df_day_sn(f, df_date)
    df_week = util.handle_original_df_week_sn(df_day)

    df = util.add_special_date(df_day, ahead_effect=5, behind_effect=2)

    data = df[["sale_cnt"]].values

    sample_num = len(data)
    train_num_3weeks = sample_num - 3 * 7
    train_num_10weeks = sample_num - 10 * 7

    ahead_period = 1

    test_data_3weeks = data[train_num_3weeks - ahead_period:]
    test_data_10weeks = data[train_num_10weeks - ahead_period:]

    true_y_3weeks = test_data_3weeks[ahead_period:, 0]
    true_y_10weeks = test_data_10weeks[ahead_period:, 0]

    record_3weeks[pid] = true_y_3weeks[:7]
    record_10weeks[pid] = true_y_10weeks[:7]

with open("input/true_value_3weeks.csv", "w") as w:
    w.write("pid,day1,day2,day3,day4,day5,day6,day7\n")
    for k, v in record_3weeks.items():
        w.write(str(k) + "," + ",".join([str(x) for x in v.tolist()]) + "\n")

with open("input/true_value_10weeks.csv", "w") as w:
    w.write("pid,day1,day2,day3,day4,day5,day6,day7\n")
    for k, v in record_10weeks.items():
        w.write(str(k) + "," + ",".join([str(x) for x in v.tolist()]) + "\n")
