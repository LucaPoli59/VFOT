import matplotlib.pyplot as plt

# import dei miei moduli
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import timeit
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from numba import jit
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.cluster.vq import whiten
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from tensorflow.python import keras

import trading_system.option_chain_functions as option_preparation
from data_preparation.data_collection_functions import get_feature_df, get_price
import data_preparation.features_processing_function as ft_processing
from main_dir import output_function as out
from data_preparation import split_functions as split
from main_dir import constant as const

from pandas.tseries.offsets import BDay as BDay


def autocorr_plot(dataframe, title):
    fig, ax = plt.subplots()
    plt.title(title)
    pd.plotting.autocorrelation_plot(dataframe)
    return fig, ax


def corr_plot(dataframe, title):
    corr_matrix = dataframe.corr()
    fig, ax = plt.subplots()
    plt.title(title)
    sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                annot=True, cmap='coolwarm', ax=ax)
    return fig, ax


def _diagnostics_diagram(dataframe, title):
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for column, color in zip(dataframe.columns, color_list):
        plt.suptitle(title + column, fontsize=18, fontweight='bold')
        plt.subplot(1, 2, 1)

        plt.hist(dataframe[column], density=True, color=color)
        plt.title('Istogramma ritorni ')

        plt.subplot(1, 2, 2)
        dataframe[column].plot.density(color=color)
        plt.title('Densità ritorni ')
        plt.tight_layout()
        plt.show()


def get_alt_target(feature_df):

    df_out = feature_df.copy()
    df_out.insert(0, column=const.TARGET_COLUMN_NAME,
                  value=(feature_df['RV'].shift(-const.TRADING_HOLDING_PERIOD) - feature_df['IV']) /
                        feature_df['IV'])

    df_out = df_out.dropna()
    return df_out


# impostazioni generali di plotting
plt.style.use('seaborn')
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['figure.dpi'] = 125
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.max_open_warning'] = 30

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""cmp_df = ft_processing.get_target_forecast(get_feature_df())[[const.TARGET_COLUMN_NAME] + const.MainFeatures.all]
cmp_df["RV_Future"] = cmp_df[const.MainFeatures.realized_volatility].shift(-const.PREDICTION_STEP)
cmp_df = cmp_df.dropna()

cmp_df.plot()
cmp_df.plot(subplots=True)
plt.show()"""

feature_df = get_feature_df()

target = np.log(feature_df["RV"].shift(-10) / feature_df['IV']).dropna()
target.name = "current target {log ^RV / IV}"


old_target = np.log(feature_df["RV"].shift(-10) / feature_df['RV']).dropna()
old_target.name = "log ^RV / RV"

target_v2 = (feature_df["RV"].shift(-10) - feature_df['IV']).dropna()
target_v2.name = "^RV - IV"

class_target = ft_processing.get_target_classification(feature_df)
print(class_target['Vola_Position'].tail(150))
"""
series_list = [target, old_target, target_v2]

for serie in series_list:

    out.simple_plot(serie.to_frame(), str(serie.name))
    autocorr_plot(serie, str(serie.name) + " auto-correlation")
    corr_plot(ft_processing.stationary(feature_df).join(serie), str(serie.name) + " correlation")

adfuller_df = pd.DataFrame(columns=['stat', '1%', '5%', '10%'], index=[serie.name for serie in series_list],
                           data=[(lambda x: [x[0]] + list(x[4].values()))(adfuller(serie)) for serie in series_list])

print("adfuller test:\n", adfuller_df, "\n\n")


ljung_box_df = pd.concat({serie.name: acorr_ljungbox(serie).transpose() for serie in series_list},
                         keys=[serie.name for serie in series_list])
ljung_box_df.columns = ["lag_" + str(column) for column in ljung_box_df.columns]
print("ljung box test:\n", ljung_box_df, "\n\n")


plt.show()"""
