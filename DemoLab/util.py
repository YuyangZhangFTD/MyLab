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
    df = original_df.copy()
    df["daytime"] = df["order_date"].apply(lambda x: str(x)) \
        .apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
    df = df.groupby("daytime", as_index=False).sum()
    df = pd.merge(time_df, df, how="left", on="daytime")
    df = df.drop("order_date", axis=1)
    df["sale_cnt"] = df["sale_cnt"].fillna(0)
    df["average_price"] = (df["pay_amt"] / df["sale_cnt"]) \
        .fillna(method="ffill").fillna(method="bfill")
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
    df["average_price"] = (
        df["pay_amt"] / df["sale_cnt"]
    ).fillna(method="ffill").fillna(method="bfill")
    df["log_average_price"] = df["average_price"].apply(lambda x: max(np.log10(x), 0))
    df["log_sale_cnt"] = df["log_sale_cnt"].apply(lambda x: x/7)
    return df
