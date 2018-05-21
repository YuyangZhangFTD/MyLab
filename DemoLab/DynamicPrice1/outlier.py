"""
    There are 4 kinds of outliers:
        1. Additive Outlier
        2. Innovational Outlier
        3. Level Shift Outlier
        4. Temporary Change Outlier
"""
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

k = 6