import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import forest

train = pd.read_csv("data.csv")
test = pd.read_csv("question.csv")

year1 = train[train.YEAR == 1]
year2 = train[train.YEAR == 2]
year3 = train[train.YEAR == 3]

year1_mean, year1_std = np.mean(year1.values[:, -1]), np.std(year1.values[:, -1])
year2_mean, year2_std = np.mean(year2.values[:, -1]), np.std(year2.values[:, -1])
year3_mean, year3_std = np.mean(year3.values[:, -1]), np.std(year3.values[:, -1])

year1 = year1[
    (year1.ROEy > year1_mean - 3 * year1_std) &
    (year1.ROEy < year1_mean + 3 * year1_std)
    ]
year2 = year2[
    (year2.ROEy > year2_mean - 3 * year2_std) &
    (year2.ROEy < year2_mean + 3 * year2_std)
    ]
year3 = year3[
    (year3.ROEy > year3_mean - 3 * year3_std) &
    (year3.ROEy < year3_mean + 3 * year3_std)
    ]

year1_mean, year1_std = np.mean(year1.values[:, -1]), np.std(year1.values[:, -1])
year2_mean, year2_std = np.mean(year2.values[:, -1]), np.std(year2.values[:, -1])
year3_mean, year3_std = np.mean(year3.values[:, -1]), np.std(year3.values[:, -1])

df = [year1, year2, year3]
train = pd.concat(df)

train["roe_1"] = train["ROE"].apply(lambda x: 1 if x < 0 else 0)
train["roe_2"] = train["ROE"].apply(lambda x: 1 if 0 <= x < 0.6 else 0)
train["roe_3"] = train["ROE"].apply(lambda x: 1 if x >= 0.5 else 0)

# same distribution
train = train.sample(frac=1.0)
df_test, df_train = train.iloc[:200], train.iloc[200:]

# RF
model = forest.RandomForestRegressor(n_estimators=5, max_depth=5)
model.fit(df_train.values[:, :-1], df_train.values[:, -1])
hat_y = model.predict(df_test.values[:, :-1])

# test
res = np.abs(hat_y - df_test.values[:, -1])
res.sort()
print(np.median(res))

'''
n_estimator     max_depth       medape        
5               5               0.0460633 
10              5               0.0430469
15              5               0.0419037
20              5               0.0405280
25              5               0.0413757
5               10              0.0416917
5               15              0.0396622
3               8               0.0375701
'''

