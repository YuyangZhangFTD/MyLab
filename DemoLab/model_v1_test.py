import pandas as pd
import datetime as dt
import numpy as np
from statsmodels import api as sm
import util


def _generate_model(param_w):
    def model(param_x):
        return np.dot(param_x, param_w)

    return model


def _predict_util(param_model, param_x, param_periods, upper_bound=999999999, lower_bound=0):
    y = []
    for _i in range(param_periods):
        hat = param_model(param_x[_i])
        y.append(
            hat if lower_bound < hat < upper_bound else (upper_bound if hat > upper_bound else lower_bound)
        )
        param_x[_i + 1, 0] = y[_i]
    return y


if __name__ == "__main__":

    period = 1
    gap = 14

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

    record_hat = []
    record_exp = []

    for i in range(316):
        if i in wrong_product:
            continue
        f = pd.read_csv("input/" + str(i) + ".csv")
        df_day = util.handle_original_df_day_sn(f, df_date)
        df_week = util.handle_original_df_week_sn(df_day)

        data = df_day[["log_sale_cnt", "log_average_price"]].values

        num_var = 2
        x = np.zeros((len(data) - 2, num_var))
        y = np.ones((len(data) - 2, 1))

        # x[:, 0] = data[1:-1, 0]  # d_{t-1}
        x[:, 0] = data[2:, 1]  # p_{t}
        x[:, 1] = data[1:-1, 1]  # p_{t-1}

        y = data[2:, 0]  # d_{t}

        data_num = len(y)
        train_num = data_num - period - gap
        train_x = x[:train_num, :]
        test_x = x[train_num:, :]
        train_y = y[:train_num]
        test_y = y[train_num:]

        max_y = np.max(train_y)
        # min_x = np.min(train_x)
        min_y = 0

        result = sm.OLS(train_y, train_x).fit()
        w = result.params.reshape(num_var, 1)

        func = _generate_model(w)
        y_true = np.array(test_y[:period]).reshape(period, 1)
        y_hat = np.array(_predict_util(func, test_x, period, max_y, min_y)).reshape(period, 1)
        y_exp = np.dot(test_x[:period, :], w)

        for j in range(period):
            y_exp[j] = y_exp[j] if (min_y < y_exp[j] < max_y) else (max_y if y_exp[j] > max_y else min_y)

        y1 = np.power(10, y_true).reshape(period, 1)
        y2 = np.power(10, y_hat).reshape(period, 1)
        y3 = np.power(10, y_exp).reshape(period, 1)

        record_hat.append(np.mean(np.abs(y1 - y2) / (y1 + y2 + np.ones((period, 1)))))
        record_exp.append(np.mean(np.abs(y1 - y3) / (y1 + y3 + np.ones((period, 1)))))

    print(np.mean(record_hat))
    print(np.mean(record_exp))
