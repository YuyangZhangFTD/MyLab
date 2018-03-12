import pandas as pd
import numpy as np
from collections import defaultdict


def _loss(array1, array2):
    return np.mean(np.abs(array1 - array2) / (array1 + array2 + 1))


# settings
# file_name_3weeks = "linear_model_v5_exp_3weeks.csv"
# file_name_10weeks = "linear_model_v5_exp_10weeks.csv"
file_name_3weeks = "log_2stage_model_v1_exp_3weeks.csv"
file_name_10weeks = "log_2stage_model_v1_exp_10weeks.csv"

# read test id list
pid_df = pd.read_csv("input/id_list.csv")
pid_list = pid_df["id"].values.tolist()  # 135 ids

# load data
df_3weeks = pd.read_csv("input/" + file_name_3weeks, index_col=0)
df_10weeks = pd.read_csv("input/" + file_name_10weeks, index_col=0)
true_value_3weeks = pd.read_csv("input/true_value_3weeks.csv", index_col=0)
true_value_10weeks = pd.read_csv("input/true_value_10weeks.csv", index_col=0)

loss = defaultdict(list)

for pid in pid_list:

    loss["loss_3weeks"].append(_loss(df_3weeks.loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["loss_10weeks"].append(_loss(df_10weeks.loc[pid].values, true_value_10weeks.loc[pid].values))

    print("=" * 20 + str(pid) + "=" * 20)
    for k, v in loss.items():
        print(k + " ==> " + str(v[-1]))

    print(df_3weeks.loc[pid].values)
    print(true_value_3weeks.loc[pid].values)
    print(_loss(df_3weeks.loc[pid].values, true_value_3weeks.loc[pid].values))

    print(df_10weeks.loc[pid].values)
    print(true_value_10weeks.loc[pid].values)
    print(_loss(df_10weeks.loc[pid].values, true_value_10weeks.loc[pid].values))

print({
    k: np.mean(v)
    for k, v in loss.items()
})

# # jiang yu ping
# df_list_exp1_arima = [pd.read_csv("input/jiangyuping/model_pred_l1_arima1_" + str(j) + ".csv") for j in range(8)]
# df_list_exp2_arima = [pd.read_csv("input/jiangyuping/model_pred_l1_arima2_" + str(j) + ".csv") for j in range(8)]
#
# # li jing
