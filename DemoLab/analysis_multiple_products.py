import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import util
import itertools

if __name__ == "__main__":
    # =============================== settings ===============================
    """
    Parameters:
        category_id:
            the l3 category id, from 0 to 67
        category:
            a list of products, which belongs to same category
    """
    # use category id
    category_id = 55

    # get product ids
    f = pd.read_csv("input/categories_l3.csv")
    category = [str(i) for i in f[f.l3_gds_group_cd == category_id]["product_id"]]

    # for test
    # category = [str(i) for i in [85, 83, 84, 285]]
    # category = [str(i) for i in [8, 9, 10, 11, 252]]

    # ============================== processing ==============================
    # time index
    start_date = dt.datetime.strptime("2017-06-04", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2017-12-31", "%Y-%m-%d")
    df_date = util.generate_time_index_df(start_date, end_date)

    df_all = [pd.read_csv("input/" + x + ".csv") for x in category]

    df_day_all = [util.handle_original_df_day_sn(x, df_date) for x in df_all]
    df_week_all = [util.handle_original_df_week_sn(x) for x in df_day_all]

    # =============================== plot ===================================
    num_products = len(category)
    line_color = ["b", "g", "r", "y", "c", "k", "m"]
    line_marker = [".", "*", "o", "x", "^", "v", "<", ">"]
    line_kind = ["-", ":"]
    line = [
        [x[0] + x[1] + k for x in itertools.product(line_marker, line_color)]
        for k in line_kind
    ]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    line1 = []
    for i in range(num_products):
        line1.append(
            ax1.plot(df_day_all[i]["daytime"], df_day_all[i]["log_sale_cnt"], line[0][i])
        )
    ax1.set_title("product_id " + ",".join(category) + " per day")
    ax1.set_xlabel("time")
    ax1.set_ylabel("log_sale_cnt")
    ax1.legend(
        [x[0] for x in line1],
        [x + " log sale cnt" for x in category],
        loc="upper left"
    )

    ax2 = ax1.twinx()
    line2 = []
    for i in range(num_products):
        line2.append(
            ax2.plot(df_day_all[i]["daytime"], df_day_all[i]["log_average_price"], line[1][i])
        )
    ax2.set_ylabel('log_average_price')
    ax2.legend([x[0] for x in line2], category, loc='upper right')
    ax2.legend(
        [x[0] for x in line2],
        [x + " log average price" for x in category],
        loc="upper right"
    )

    ax3 = fig.add_subplot(212)
    line3 = []
    for i in range(num_products):
        line3.append(
            ax3.plot(df_week_all[i]["week"], df_week_all[i]["log_sale_cnt"], line[0][i])
        )
    ax3.set_title("product_id " + ",".join(category) + " per week")
    ax3.set_xlabel("week")
    ax3.set_ylabel("log_sale_cnt")
    ax3.legend([x[0] for x in line3], category, loc="upper left")
    ax3.legend(
        [x[0] for x in line3],
        [x + " log sale cnt" for x in category],
        loc="upper left"
    )

    ax4 = ax3.twinx()
    line4 = []
    for i in range(num_products):
        line4.append(
            ax4.plot(df_week_all[i]["week"], df_week_all[i]["log_average_price"], line[1][i])
        )
    ax4.set_ylabel('log_average_price')
    ax4.legend(
        [x[0] for x in line4],
        [x + " log average price" for x in category],
        loc="upper right"
    )

    plt.show()
