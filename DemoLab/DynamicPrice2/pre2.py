import util
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


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

original_data = original_data[original_data.daytime > "20151231"]

original_data = original_data[original_data.pay_amt < 5]

original_data["pid"] = original_data["pid"].astype(str)

original_data["individul_mean_pay"] = (
    original_data["pay_amt"] / original_data["sale_cnt"]
)
original_data["max_pay"] = original_data["individul_average_price"]
original_data["min_pay"] = original_data["individul_average_price"]
original_data["mean_pay"] = original_data["individul_average_price"]
original_data["std_pay"] = original_data["individul_average_price"]
original_data["max_price"] = original_data["sale_price"]
original_data["min_price"] = original_data["sale_price"]
original_data["mean_price"] = original_data["sale_price"]
original_data["std_price"] = original_data["sale_price"]
original_data["max_low_price"] = original_data["low_price"]
original_data["min_low_price"] = original_data["low_price"]
original_data["mean_low_price"] = original_data["low_price"]
original_data["std_low_price"] = original_data["low_price"]
original_data["max_limit_price"] = original_data["limit_price"]
original_data["min_limit_price"] = original_data["limit_price"]
original_data["mean_limit_price"] = original_data["limit_price"]
original_data["std_limit_price"] = original_data["limit_price"]

data = original_data.groupby(["daytime", "pid"], as_index=False).agg(
    {
        "sale_cnt": np.sum,
        "pay_amt": np.sum,
        "max_pay": np.max,
        "min_pay": np.min,
        "mean_pay": np.mean,
        "std_pay": np.std,
        "max_price": np.max,
        "min_price": np.min,
        "mean_price": np.mean,
        "std_price": np.std,
        "max_low_price": np.max,
        "min_low_price": np.min,
        "mean_low_price": np.mean,
        "std_low_price": np.std,
        "max_limit_price": np.max,
        "min_limit_price": np.min,
        "mean_limit_price": np.mean,
        "std_limit_price": np.std,
    }
)
data = data.fillna(0)  # std is NaN

data["average_price"] = data["sale_amt"] / data["sale_cnt"]
data["average_pay"] = data["pay_amt"] / data["sale_cnt"]

# time index
start_date = dt.datetime.strptime("2016-01-04", "%Y-%m-%d")
end_date = dt.datetime.strptime("2018-04-30", "%Y-%m-%d")
timedf = util.generate_time_index_df(start_date, end_date)

list_dfs = []
for pid in pid_list:
    list_dfs.append(pd.merge(timedf, data[data.pid == pid], how="left", on="daytime"))
