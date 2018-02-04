import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


if __name__ == "__main__":
    # =============================== settings ===============================
    single_product = "4"
    # single_product = "5"
    # single_product = "185"
    # single_product = "191"

    # test
    # single_product = "110"

    # ============================== processing ==============================
    # time index
    start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    end = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
    days = (end - start_date).days
    day_list = [start_date + dt.timedelta(i) for i in range(days)]
    df_date = pd.DataFrame(data=day_list, index=range(days), columns=["daytime"])
    df_date["weektime"] = df_date["daytime"].apply(lambda x: start_date + dt.timedelta((x-start_date).days // 7 * 7))

    # TODO: use df_date in sn and jd df

    # sn data per day
    # read file
    original_df = pd.read_csv("input/" + single_product + ".csv")[["order_date", "sale_cnt", "pay_amt"]]

    # sale amount every day
    df_sn = original_df.groupby("order_date", as_index=False).sum()
    df_sn = df_sn[df_sn["order_date"] > 20170531]

    # average price for each day
    df_sn["average_price"] = df_sn["pay_amt"] / df_sn["sale_cnt"]

    # add time column
    df_sn["order_date"] = df_sn["order_date"].apply(lambda x: str(x)) \
        .apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))

    # log sale cnt
    df_sn["log_sale_cnt"] = df_sn["sale_cnt"].apply(lambda x: np.log10(x))

    # log average price
    df_sn["log_average_price"] = df_sn["average_price"].apply(lambda x: np.log10(x))

    # data of every weeks, same as those of each day
    week_flag = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    df_sn_week = df_sn[df_sn.order_date > week_flag]
    df_sn_week["week"] = df_sn_week["order_date"].apply(lambda x: _cal_week(x))
    df_sn_week = df_sn_week.groupby("week", as_index=False).sum()
    df_sn_week["average_price"] = df_sn_week["pay_amt"] / df_sn_week["sale_cnt"]
    df_sn_week["log_average_price"] = df_sn_week["average_price"].apply(lambda x: np.log10(x))

    # jd price data per day
    jd_history = ''
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
        jd_history = [[dt.datetime.strptime(y[0][:10], "%Y-%m-%d"), float(y[1])]
                      for y in [x.split("->") for x in jd_history.split(":::")]]
        df_jd = pd.DataFrame(data=jd_history, index=range(len(jd_history)), columns=["time", "price"]) \
            .groupby("time", as_index=False).mean()
        df_jd["log_price"] = df_jd["price"].apply(lambda x: np.log10(x))

        # jd price data per week
        week_flag = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
        df_jd_week = df_jd[df_jd.time > week_flag]
        df_jd_week["week"] = df_jd_week["time"].apply(lambda x: _cal_week(x))
        df_jd_week = df_jd_week.groupby("week", as_index=False).mean()
        df_jd_week["log_price"] = df_jd_week["price"].apply(lambda x: np.log10(x))

    # =============================== plot ===================================
    fig = plt.figure()
    # per day
    ax1 = fig.add_subplot(221)
    ax1.set_title("product_id " + single_product + " per day")
    ax1.plot(df_sn["order_date"], df_sn["sale_cnt"], "r.-")
    ax1.plot(df_sn["order_date"], [0 for __ in range(len(df_sn.index))], "y")
    ax1.set_xlabel('time')
    ax1.set_ylabel("sale_cnt")
    ax1.legend(loc='upper center')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(df_sn["order_date"], df_sn["average_price"], "b.-")
    if jd_history:
        ax2.plot(df_jd["time"], df_jd["price"], "g.-")
    ax2.set_ylabel('average_price')
    ax2.legend(loc='upper right')

    ax3 = fig.add_subplot(223)
    ax3.set_title(
        "product_id " +
        single_product +
        " with logarithmic processing per day")
    ax3.plot(df_sn["order_date"], df_sn["log_sale_cnt"], "r.-")
    ax3.plot(df_sn["order_date"], [0 for __ in range(len(df_sn.index))], "y")
    ax3.set_xlabel('time')
    ax3.set_ylabel("log_sale_cnt")
    ax3.legend(loc='upper center')

    ax4 = ax3.twinx()
    ax4.plot(df_sn["order_date"], df_sn["log_average_price"], "b.-")
    if jd_history:
        ax4.plot(df_jd["time"], df_jd["log_price"], "g.-")
    ax4.set_ylabel('log_average_price')
    ax4.legend(loc='upper right')

    # per week
    ax5 = fig.add_subplot(222)
    ax5.set_title("product_id " + single_product + " per week")
    ax5.plot(df_sn_week["week"], df_sn_week["sale_cnt"], "r.-")
    ax5.plot(df_sn_week["week"], [0 for __ in range(len(df_sn_week.index))], "y")
    ax5.set_xlabel('time')
    ax5.set_ylabel("sale_cnt")
    ax5.legend(loc='upper center')

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
    ax7.plot(df_sn_week["week"], [0 for __ in range(len(df_sn_week.index))], "y")
    ax7.set_xlabel('time')
    ax7.set_ylabel("log_sale_cnt")
    ax7.legend(loc='upper center')

    ax8 = ax7.twinx()  # this is the important function
    ax8.plot(df_sn_week["week"], df_sn_week["log_average_price"], "b.-")
    if jd_history:
        ax8.plot(df_jd_week["week"], df_jd_week["log_price"], "g.-")
    ax8.set_ylabel('log_average_price')
    ax8.legend(loc='upper right')

    plt.show()
