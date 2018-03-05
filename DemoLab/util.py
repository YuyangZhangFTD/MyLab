import pandas as pd
import datetime as dt
import numpy as np


def generate_time_index_df(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    :return:
    """
    days = (end_date - start_date).days
    day_list = [start_date + dt.timedelta(i) for i in range(days)]
    df_date = pd.DataFrame(
        data=day_list,
        index=range(days),
        columns=["daytime"])
    df_date["week"] = df_date["daytime"].apply(
        lambda x: start_date + dt.timedelta((x - start_date).days // 7 * 7)
    )
    return df_date


def handle_original_df_day_sn(original_df, time_df):
    """

    :param original_df:
    :param time_df:
    :return:
    """
    df = original_df[["order_date", "sale_cnt", "pay_amt"]]
    df["daytime"] = df.loc[:, "order_date"].apply(lambda x: str(x)) \
        .apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
    df = df.groupby("daytime", as_index=False).sum()
    df = pd.merge(time_df, df, how="left", on="daytime")
    df = df.drop("order_date", axis=1)
    df["sale_cnt"] = df["sale_cnt"].fillna(0)
    df["average_price"] = df["pay_amt"] / df["sale_cnt"]
    df["average_price"] = df["average_price"].fillna(df["average_price"].max())
    df["log_sale_cnt"] = df["sale_cnt"].apply(lambda x: max(np.log10(x), 0))
    df["log_average_price"] = df["average_price"].apply(
        lambda x: max(np.log10(x), 0)
    )
    return df


def handle_original_df_week_sn(day_df):
    """

    :param day_df:
    :return:
    """
    df = day_df.copy()
    df = df.groupby("week", as_index=False).sum()
    df["average_price"] = df["pay_amt"] / df["sale_cnt"]
    df["average_price"] = df["average_price"].fillna(df["average_price"].max())
    df["log_average_price"] = df["average_price"].apply(lambda x: max(np.log10(x), 0))
    df["log_sale_cnt"] = df["log_sale_cnt"].apply(lambda x: x / 7)
    return df


def generate_time_series(df, column_name, index_name):
    return pd.Series(df[column_name].values.copy(), index=df[index_name].values)


def add_special_date(df, ahead_effect=7, behind_effect=3):
    df["special_date"] = 0
    df["ahead_special_date"] = 0
    df["behind_special_date"] = 0

    df.loc[df.daytime == "20170618", "special_date"] = 1
    df.loc[df.daytime == "20170818", "special_date"] = 1
    df.loc[df.daytime == "20171111", "special_date"] = 1
    df.loc[df.daytime == "20171212", "special_date"] = 1

    # ahead effect
    effect = behind_effect
    for i in range(len(df)):
        if df.loc[i, "special_date"] == 1 or effect < behind_effect:
            df.loc[i, "behind_special_date"] = 1 if df.loc[i, "special_date"] != 1 else 0
            effect = effect - 1 if effect > 0 else behind_effect

    # behind effect
    effect = ahead_effect
    for i in range(len(df)-1, -1, -1):
        if df.loc[i, "special_date"] == 1 or effect < ahead_effect:
            df.loc[i, "ahead_special_date"] = 1 if df.loc[i, "special_date"] != 1 else 0
            effect = effect - 1 if effect > 0 else ahead_effect

    return df


def _aic_outlier(param_ts, param_n, param_alpha):
    return param_alpha * param_n + np.log(param_ts.std())


def eliminate_outlier_ts(ts, alpha, method="mean"):
    ts_zero_mean = ts - ts.mean()
    old_aic = _aic_outlier(ts, 0, alpha)
    n_outlier = 0
    aic_history = []
    while True:
        outlier_index = ts_zero_mean.abs().argmax()
        if outlier_index in ts_zero_mean.index:
            outlier_value = ts_zero_mean[outlier_index]
            ts_zero_mean[outlier_index] = None
            n_outlier += 1
            if _aic_outlier(ts_zero_mean, n_outlier, alpha) > old_aic:
                break
            else:
                ts[outlier_index] = None
                old_aic = _aic_outlier(ts_zero_mean, n_outlier, alpha)
                aic_history.append((outlier_index, outlier_value))
        else:
            break

    if method == "mean":
        ts = ts.fillna(ts.mean())
    else:
        print("Wrong method for fill missing value")

    return ts, aic_history


def eliminate_outlier_df(df, alpha, column_name, index_name):
    ts = generate_time_series(df, column_name, index_name)
    ts, aic_history = eliminate_outlier_ts(ts, alpha)
    return pd.merge(
        df, ts.reset_index(name="sale_cnt_no_outlier"),
        how="inner", left_on=index_name, right_on="index"
    ), aic_history
