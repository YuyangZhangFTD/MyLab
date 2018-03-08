import pandas as pd


pid_df = pd.read_csv("input/id_list.csv")
id_list = pid_df["id"].values.tolist()  # 135 ids

d = 1

# jiang yu ping
df_list_exp1_arima = [pd.read_csv("input/jiangyuping/model_pred_l1_arima1_" + str(j) + ".csv") for j in range(8)]
df_list_exp2_arima = [pd.read_csv("input/jiangyuping/model_pred_l1_arima2_" + str(j) + ".csv") for j in range(8)]

# li jing
