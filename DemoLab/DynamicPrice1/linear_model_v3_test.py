import util
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm

if __name__ == "__main__":

    test_week_num = 3
    is_log = False

    pid_df = pd.read_csv("input/id_list.csv")
    id_list = pid_df["id"].values.tolist()  # 135 ids

    wrong_product = [
        33, 49, 52, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74,
        89, 100, 107, 120, 149, 159, 203,
        218, 223, 312, 313, 150
    ]

    start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
    df_date = util.generate_time_index_df(start_date, end_date)

    if is_log:
        file_name = "input/log_linear_model_v3_exp_" + str(test_week_num) + "weeks.csv"
    else:
        file_name = "input/linear_model_v3_exp_" + str(test_week_num) + "weeks.csv"

    with open(file_name, "w") as file:

        file.write("pid,day1,day2,day3,day4,day5,day6,day7\n")

        for pid in id_list:

            if pid in wrong_product:
                continue

            f = pd.read_csv("input/" + str(pid) + ".csv")
            df_day = util.handle_original_df_day_sn(f, df_date)
            df_week = util.handle_original_df_week_sn(df_day)

            df = util.add_special_date(df_day, ahead_effect=5, behind_effect=2)

            if is_log:
                data = df[[
                    "log_sale_cnt",
                    "log_average_price",
                    "special_date",
                    "ahead_special_date",
                    "behind_special_date"
                ]].values
            else:
                data = df[[
                    "sale_cnt",
                    "average_price",
                    "special_date",
                    "ahead_special_date",
                    "behind_special_date"
                ]].values

            sample_num = len(data)
            train_num = sample_num - test_week_num * 7

            ahead_period = 1

            train_data = data[:train_num]
            test_data = data[train_num - ahead_period:]

            var_num = 5

            x = np.zeros((train_num - ahead_period, var_num))

            x[:, 0] = train_data[ahead_period:, 1]  # p_{t}
            x[:, 1] = train_data[ahead_period - 1: -1, 0]  # d_{t-1}
            x[:, 2] = train_data[ahead_period - 1: -1, 1]  # p_{t-1}
            x[:, 3] = np.power(train_data[ahead_period:, 1], 2)
            x[:, 4] = np.power(train_data[ahead_period:, 1], 3)

            y = np.zeros((train_num - ahead_period, 1))
            y = train_data[ahead_period:, 0]

            result = sm.OLS(y, x).fit()
            print(result.summary())

            # predict
            w = result.params.reshape(var_num, 1)
            hat_y = np.dot(x, w)

            test_x = np.zeros((sample_num - train_num, var_num))
            test_x[:, 0] = test_data[ahead_period:, 1]  # p_{t}
            test_x[:, 1] = test_data[ahead_period - 1:sample_num - train_num, 0]  # d_{t-1}
            test_x[:, 2] = test_data[ahead_period - 1:sample_num - train_num, 1]  # p_{t-1}
            test_x[:, 3] = np.power(test_data[ahead_period:, 1], 2)
            test_x[:, 4] = np.power(test_data[ahead_period:, 1], 3)

            test_y = np.zeros((sample_num - train_num, 1))
            test_y[:, 0] = test_data[ahead_period:, 0]
            predict_y = np.dot(test_x, w)

            if is_log:
                calculate_y = np.power(10, predict_y[:7].reshape(1, 7))
            else:
                calculate_y = predict_y[:7].reshape(1, 7)

            s = str(pid) + "," + ",".join([str(x) if x > 0 else "0" for x in calculate_y[0].tolist()]) + "\n"
            file.write(s)
