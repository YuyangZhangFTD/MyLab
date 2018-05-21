import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from matplotlib import pyplot as plt
from statsmodels.graphics.api import qqplot
import util
from collections import defaultdict

if __name__ == "__main__":

    period = 1
    gap = 14

    wrong_product = [
        33, 49, 52, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74,
        89, 100, 107, 120, 149, 159, 203,
        218, 223, 312, 313, 150
    ]

    start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
    df_date = util.generate_time_index_df(start_date, end_date)

    record = defaultdict(list)
    not_converge_count = 0

    for i in range(316):
        if i in wrong_product:
            continue

        f = pd.read_csv("input/" + str(i) + ".csv")
        df_day = util.handle_original_df_day_sn(f, df_date)
        df_week = util.handle_original_df_week_sn(df_day)

        ts_sales = df_day[["daytime", "sale_cnt"]].set_index("daytime")
        ts_log_sales = df_day[["daytime", "log_sale_cnt"]].set_index("daytime")

        try:
            (t, pvalue, lag, *_) = ts.adfuller(ts_log_sales.values.T[0])
        except np.linalg.linalg.LinAlgError:
            print("product "+str(i)+" SVD did not converge")
            not_converge_count += 1

        record[lag].append((i, pvalue))

    print(not_converge_count)
    print(len(record[0]))
    print(len(record[1]))
    print(sum(record))
