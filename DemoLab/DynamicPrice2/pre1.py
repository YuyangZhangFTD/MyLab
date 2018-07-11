import util
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
-- 农夫山泉矿泉水
104062167
104062165
104062173
126448576
104062171
-- 怡宝矿泉水
121614035
121614188
121615083
121676928
121677095
-- 康师傅矿泉水
104131265
104131266
104131264
"""
pid_list = [
    "104062167",
    "104062165",
    "104062173",
    "126448576",
    "104062171",
    "121614035",
    "121614188",
    "121615083",
    "121676928",
    "121677095",
    "104131265",
    "104131266",
    "104131264",
]

original_data = pd.read_csv("input2/order_sale.txt", delimiter="\t")

original_data.columns = [
    "daytime",
    "time",
    "pid",
    "sale_price",
    "sale_amt",
    "sale_cnt",
    "pay_amt",
    "low_price",
    "lower_price_amt",
    "discount_amt",
    "limit_price",
]

original_data = original_data[
    [
        "daytime",
        "time",
        "pid",
        "sale_price",  # 销售单价
        "sale_amt",  # 销售金额
        "sale_cnt",
        "pay_amt",  # 付款金额
        "low_price",  # 底价
        "limit_price",  # 集团限价
    ]
]

original_data["daytime"] = (
    original_data.loc[:, "daytime"]
    .astype(str)
    .apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
)

# clearness
original_data["pid"] = original_data["pid"].astype(str)
original_data = original_data[original_data.daytime > "20151231"]

# # get clear dataa
# data = original_data[
#     (original_data.pid != "659182759") &
#     (original_data.pid != "104062166") &
#     (original_data.pid != "163703087") &
#     (original_data.pid != "125081686") &
#     (original_data.pid != "164570891") &
#     (original_data.pid != "688396007") &
#     (original_data.pid != "153148786")
#     ]

data_daytime_pid = original_data.groupby(["daytime", "pid"], as_index=False).agg({
        "sale_cnt": np.sum,
        "sale_amt": np.sum,
        "pay_amt": np.sum,
        "low_price": np.mean,
        "limit_price": np.mean,
})

data_all = original_data
data_all["average_price"] = (data_all["sale_amt"] / data_all["sale_cnt"]).apply(lambda x: round(x, 1))
data_all["average_pay"] = (data_all["pay_amt"] / data_all["sale_cnt"]).apply(lambda x: round(x, 1))

data_all_price = data_all.groupby(["pid", "average_price"], as_index=False).agg({
    # "sale_amt": np.sum,
    "sale_cnt": np.sum
})
# data_all_price["average_price"] = data_all_price["sale_amt"] / data_all_price["sale_cnt"]

data_all_pay = data_all.groupby(["pid", "average_pay"], as_index=False).agg({
    # "pay_amt": np.sum,
    "sale_cnt": np.sum
})
# data_all_pay["average_pay"] = data_all_pay["pay_amt"] / data_all_pay["sale_cnt"]

# data = original_data.groupby(["daytime", "pid"], as_index=False).sum()

data_daytime_pid["average_price"] = (
    data_daytime_pid["sale_amt"] / data_daytime_pid["sale_cnt"]
)
data_daytime_pid["average_pay"] = (
    data_daytime_pid["pay_amt"] / data_daytime_pid["sale_cnt"]
)

sum_df = data_daytime_pid.groupby("daytime", as_index=False).sum()

# time index
start_date = dt.datetime.strptime("2016-01-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2018-04-30", "%Y-%m-%d")
time_df = util.generate_time_index_df(start_date, end_date)

list_dfs = []
list_dfs_price = []
list_dfs_pay = []
for pid in pid_list:
    list_dfs.append(
        pd.merge(
            time_df,
            data_daytime_pid[data_daytime_pid.pid == pid],
            how="left",
            on="daytime",
        )
    )
    list_dfs_price.append(data_all_price[data_all_price.pid == pid])
    list_dfs_pay.append(data_all_pay[data_all_pay.pid == pid])

# sum_df = pd.merge(time_df, sum_df, how="left", on="daytime")

# plot
line_color = ["b", "g", "r", "y", "c", "k", "m"]
line_marker = [".", "*", "o", "x", "^", "v", "<", ">"]
line_kind = ["-", ":"]

for i in range(len(pid_list)):

    fig = plt.figure(pid_list[i], dpi=150)

    ax1 = fig.add_subplot(311)
    ax1.plot(list_dfs[0]["daytime"], list_dfs[i]["sale_cnt"], "b-.")
    ax1.set_xlabel("time")
    ax1.set_ylabel("sales")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(list_dfs[i]["daytime"], list_dfs[i]["average_pay"], "r-.")
    ax2.set_ylabel("price")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(312)
    ax3.plot(list_dfs[i]["daytime"], list_dfs[i]["average_pay"], "r-.")
    ax3.plot(list_dfs[i]["daytime"], list_dfs[i]["average_price"], "g-.")
    ax3.plot(list_dfs[i]["daytime"], list_dfs[i]["low_price"], "y-.")
    ax3.plot(list_dfs[i]["daytime"], list_dfs[i]["limit_price"], "c-.")
    ax3.set_xlabel("time")
    ax3.set_ylabel("price")
    ax3.legend(loc="lower left")

    average_low_price = np.mean(list_dfs[i]["low_price"])
    average_limit_price = np.mean(list_dfs[i]["limit_price"])

    ax4 = fig.add_subplot(325)
    # ax4.scatter(list_dfs[i]["average_pay"], list_dfs[i]["sale_cnt"], c="r", s=2)
    ax4.scatter(list_dfs_pay[i]["average_pay"], list_dfs_pay[i]["sale_cnt"], c="r", s=0.5)
    ax4.plot([average_low_price] * 2, [0, list_dfs_pay[i]["sale_cnt"].max()], "y--")
    ax4.plot([average_limit_price] * 2, [0, list_dfs_pay[i]["sale_cnt"].max()], "c--")
    ax4.set_xlabel("average_pay")
    ax4.set_ylabel("sale_cnt")

    ax5 = fig.add_subplot(326)
    # ax5.scatter(list_dfs[i]["average_price"], list_dfs[i]["sale_cnt"], c="g", s=2)
    ax5.scatter(list_dfs_price[i]["average_price"], list_dfs_price[i]["sale_cnt"], c="r", s=0.5)
    ax5.plot([average_low_price] * 2, [0, list_dfs_price[i]["sale_cnt"].max()], "y--")
    ax5.plot([average_limit_price] * 2, [0, list_dfs_price[i]["sale_cnt"].max()], "c--")
    ax5.set_xlabel("average_price")
    ax5.set_ylabel("sale_cnt")

plt.show()
