import pandas as pd
import numpy as np
from collections import defaultdict


def _loss(array1, array2):
    return np.mean(np.abs(array1 - array2) / (array1 + array2 + 1))


# settings
file_name_3weeks = "model_pred_l1_arima_3weeks_"
file_name_10weeks = "model_pred_l1_arima_3weeks_"

# read test id list
pid_df = pd.read_csv("input/id_list.csv")
pid_list = pid_df["id"].values.tolist()  # 135 ids

# load data
true_value_3weeks = pd.read_csv("input/true_value_3weeks.csv", index_col=0)
true_value_10weeks = pd.read_csv("input/true_value_10weeks.csv", index_col=0)

pds_3weeks = pd.concat([
    pd.read_csv("input/jiangyuping/" + file_name_3weeks + str(i) + " .csv")
    for i in range(8)
], axis=0)
pds_10weeks = pd.concat([
    pd.read_csv("input/jiangyuping/" + file_name_10weeks + str(i) + " .csv")
    for i in range(8)
], axis=0)

df_3weeks = pds_3weeks.pivot_table(["pred"], index="gds_cd", columns="head")
df_10weeks = pds_3weeks.pivot_table(["pred"], index="gds_cd", columns="head")

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
