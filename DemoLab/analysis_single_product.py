import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import util

if __name__ == "__main__":
    # =============================== settings ===============================
    """
    Parameters:
        need_jd_history: 
            whether jd's data will be shown
        single_product:
            the product id, from 0 to 315
    Attention:
        wrong_product = [
            33, 49, 52, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 66,
            67, 68, 69, 70, 71, 72, 73, 74,
            89, 100, 107, 120, 149, 159, 203,
            218, 223, 312, 313, 150
        ]
    """
    need_jd_history = True
    product_id = 82

    # ============================== processing ==============================
    # time index
    start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
    df_date = util.generate_time_index_df(start_date, end_date)

    # sn data per day
    single_product = str(product_id)
    # read file
    original_df = pd.read_csv("input/" +
                              single_product +
                              ".csv")[["order_date", "sale_cnt", "pay_amt"]]

    # sale amount every day
    df_sn = original_df.groupby("order_date", as_index=False).sum()

    # add time column
    df_sn["daytime"] = df_sn["order_date"].apply(lambda x: str(x)) \
        .apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))

    # merge data df and time df
    df_sn = pd.merge(df_date, df_sn, how="left", on="daytime")
    df_sn = df_sn.drop("order_date", axis=1)
    df_sn["sale_cnt"] = df_sn["sale_cnt"].fillna(0)

    # average price for each day
    df_sn["average_price"] = (df_sn["pay_amt"] / df_sn["sale_cnt"])
    df_sn["average_price"] = df_sn["average_price"].fillna(df_sn["average_price"].max())

    # log sale cnt
    df_sn["log_sale_cnt"] = df_sn["sale_cnt"].apply(
        lambda x: max(np.log10(x), 0))

    # log average price
    df_sn["log_average_price"] = df_sn["average_price"].apply(
        lambda x: max(np.log10(x), 0))

    # data of every weeks, same as those of each day
    df_sn_week = df_sn.copy()
    df_sn_week = df_sn_week.groupby("week", as_index=False).sum()
    df_sn_week["average_price"] = (
            df_sn_week["pay_amt"] / df_sn_week["sale_cnt"]
    )
    df_sn_week["average_price"] = df_sn_week["average_price"].fillna(df_sn_week["average_price"].max())
    df_sn_week["log_average_price"] = df_sn_week["average_price"].apply(
        lambda x: max(np.log10(x), 0)
    )
    df_sn_week["log_sale_cnt"] = df_sn_week["log_sale_cnt"].apply(lambda x: x / 7)

    # read jd's data
    jd_history = ''
    if need_jd_history:
        with open("input/jd_history.csv") as jd:
            jd.readline()
            while True:
                s = jd.readline().split(",")
                if len(s) < 2:
                    print("The history data not found")
                    break
                if s[0] == single_product:
                    jd_history = s[-1][:-1]
                    break
    if jd_history:
        # jd's price history per day
        jd_history = [[dt.datetime.strptime(y[0][:10], "%Y-%m-%d"), float(y[1])]
                      for y in [x.split("->") for x in jd_history.split(":::")]]
        df_jd = pd.DataFrame(
            data=jd_history,
            index=range(len(jd_history)),
            columns=["daytime", "price"]
        ).groupby("daytime", as_index=False).mean()
        df_jd = pd.merge(df_date, df_jd, how="left", on="daytime")
        df_jd["price"] = df_jd["price"].fillna(
            method="ffill").fillna(method="bfill")
        df_jd["log_price"] = df_jd["price"].apply(
            lambda x: max(np.log10(x), 0))

        # jd price data per week
        df_jd_week = df_jd.groupby("week", as_index=False).mean()
        df_jd_week["log_price"] = df_jd_week["price"].apply(
            lambda x: max(np.log10(x), 0))

    # =============================== plot ===================================
    fig = plt.figure()
    # per day
    ax1 = fig.add_subplot(221)
    ax1.set_title("product_id " + single_product + " per day")
    ax1.plot(df_sn["daytime"], df_sn["sale_cnt"], "r.-")
    ax1.plot(df_sn["daytime"], [0 for __ in range(len(df_sn.index))], "y")
    ax1.set_xlabel("time")
    ax1.set_ylabel("sale_cnt")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(df_sn["daytime"], df_sn["average_price"], "b.-")
    if need_jd_history and jd_history:
        ax2.plot(df_jd["daytime"], df_jd["price"], "g.-")
    ax2.set_ylabel('average_price')
    ax2.legend(loc='upper right')

    ax3 = fig.add_subplot(223)
    ax3.set_title(
        "product_id " +
        single_product +
        " with logarithmic processing per day")
    ax3.plot(df_sn["daytime"], df_sn["log_sale_cnt"], "r.-")
    ax3.plot(df_sn["daytime"], [0 for __ in range(len(df_sn.index))], "y")
    ax3.set_xlabel('time')
    ax3.set_ylabel("log_sale_cnt")
    ax3.legend(loc='upper left')

    ax4 = ax3.twinx()
    ax4.plot(df_sn["daytime"], df_sn["log_average_price"], "b.-")
    if need_jd_history and jd_history:
        ax4.plot(df_jd["daytime"], df_jd["log_price"], "g.-")
    ax4.set_ylabel('log_average_price')
    ax4.legend(loc='upper right')

    # per week
    ax5 = fig.add_subplot(222)
    ax5.set_title("product_id " + single_product + " per week")
    ax5.plot(df_sn_week["week"], df_sn_week["sale_cnt"], "r.-")
    ax5.plot(df_sn_week["week"], [
        0 for __ in range(len(df_sn_week.index))], "y")
    ax5.set_xlabel('time')
    ax5.set_ylabel("sale_cnt")
    ax5.legend(loc='upper left')

    ax6 = ax5.twinx()  # this is the important function
    ax6.plot(df_sn_week["week"], df_sn_week["average_price"], "b.-")
    if jd_history:
        ax6.plot(df_jd_week["week"], df_jd_week["price"], "g.-")
    ax6.set_ylabel('average_price')
    ax6.legend(loc='upper right')

    ax7 = fig.add_subplot(224)
    ax7.set_title(
        "product_id " +
        single_product +
        " with logarithmic processing per week")
    ax7.plot(df_sn_week["week"], df_sn_week["log_sale_cnt"], "r.-")
    ax7.plot(df_sn_week["week"], [
        0 for __ in range(len(df_sn_week.index))], "y")
    ax7.set_xlabel('time')
    ax7.set_ylabel("log_sale_cnt")
    ax7.legend(loc='upper left')

    ax8 = ax7.twinx()  # this is the important function
    ax8.plot(df_sn_week["week"], df_sn_week["log_average_price"], "b.-")
    if jd_history:
        ax8.plot(df_jd_week["week"], df_jd_week["log_price"], "g.-")
    ax8.set_ylabel('log_average_price')
    ax8.legend(loc='upper right')

    plt.show()
