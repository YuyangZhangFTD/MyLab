import util
import pandas as pd
import datetime as dt
import numpy as np
from collections import defaultdict


def _loss(array1, array2):
    return np.mean(np.abs(array1 - array2) / (array1 + array2 + 1))


# read test id list
pid_df = pd.read_csv("input/id_list.csv")
pid_list = pid_df["id"].values.tolist()  # 135 ids

# load data
linear_model_exp_3weeks = [
    pd.read_csv("input/linear_model_v" + str(i) + "_exp_3weeks.csv", index_col=0)
    for i in range(1, 5)
]
linear_model_exp_10weeks = [
    pd.read_csv("input/linear_model_v" + str(i) + "_exp_10weeks.csv", index_col=0)
    for i in range(1, 5)
]

log_linear_model_exp_3weeks = [
    pd.read_csv("input/log_linear_model_v" + str(i) + "_exp_3weeks.csv", index_col=0)
    for i in range(1, 5)
]
log_linear_model_exp_10weeks = [
    pd.read_csv("input/log_linear_model_v" + str(i) + "_exp_10weeks.csv", index_col=0)
    for i in range(1, 5)
]

true_value_3weeks = pd.read_csv("input/true_value_3weeks.csv", index_col=0)
true_value_10weeks = pd.read_csv("input/true_value_10weeks.csv", index_col=0)

loss = defaultdict(list)

for pid in pid_list:

    loss["lmv1_3w"].append(_loss(linear_model_exp_3weeks[0].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["lmv2_3w"].append(_loss(linear_model_exp_3weeks[1].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["lmv3_3w"].append(_loss(linear_model_exp_3weeks[2].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["lmv4_3w"].append(_loss(linear_model_exp_3weeks[3].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["llmv1_3w"].append(_loss(log_linear_model_exp_3weeks[0].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["llmv2_3w"].append(_loss(log_linear_model_exp_3weeks[1].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["llmv3_3w"].append(_loss(log_linear_model_exp_3weeks[2].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["llmv4_3w"].append(_loss(log_linear_model_exp_3weeks[3].loc[pid].values, true_value_3weeks.loc[pid].values))
    loss["lmv1_10w"].append(_loss(linear_model_exp_10weeks[0].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["lmv2_10w"].append(_loss(linear_model_exp_10weeks[1].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["lmv3_10w"].append(_loss(linear_model_exp_10weeks[2].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["lmv4_10w"].append(_loss(linear_model_exp_10weeks[3].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["llmv1_10w"].append(_loss(log_linear_model_exp_10weeks[0].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["llmv2_10w"].append(_loss(log_linear_model_exp_10weeks[1].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["llmv3_10w"].append(_loss(log_linear_model_exp_10weeks[2].loc[pid].values, true_value_10weeks.loc[pid].values))
    loss["llmv4_10w"].append(_loss(log_linear_model_exp_10weeks[3].loc[pid].values, true_value_10weeks.loc[pid].values))
    print("=" * 20 + str(pid) + "=" * 20)
    for k, v in loss.items():
        print(k + " ==> " + str(v[-1]))

print({
    k: np.mean(v)
    for k, v in loss.items()
})

# # jiang yu ping
# df_list_exp1_arima = [pd.read_csv("input/jiangyuping/model_pred_l1_arima1_" + str(j) + ".csv") for j in range(8)]
# df_list_exp2_arima = [pd.read_csv("input/jiangyuping/model_pred_l1_arima2_" + str(j) + ".csv") for j in range(8)]
#
# # li jing
