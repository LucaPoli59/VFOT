import datetime

import matplotlib.pyplot as plt

# import dei miei moduli
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import timeit
import json
from pytrends.request import TrendReq

import statsmodels.tsa.seasonal
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from numba import jit
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel, RFE
from scipy.cluster.vq import whiten
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

import trading_system.option_chain_functions as option_preparation
from data_preparation.ScalerDictionaries import ScalerDictionaryM
from data_preparation.data_collection_functions import get_feature_df, get_price
import data_preparation.features_processing_function as ft_processing
from main_dir import output_function as out
from data_preparation import split_functions as split
from main_dir import constant as const

from pandas.tseries.offsets import BDay as BDay


def autocorr_plot(dataframe):
    for column in dataframe.columns:
        fig, ax = plt.subplots()
        plt.suptitle(column)
        pd.plotting.autocorrelation_plot(dataframe[column], ax=ax)


def decomposing_plot(series):
    statsmodels.tsa.seasonal.seasonal_decompose(series, model="additive", period=1).plot()  # o multiplicative
    """stl = stl.fit()
    stl.plot()"""

def dataframe_plot(dataframe):
    for column in dataframe.columns:
        fig, ax = plt.subplots()
        plt.suptitle(column)
        ax.plot(dataframe[column])

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


# impostazioni generali di plotting
plt.style.use('seaborn')
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['figure.dpi'] = 125
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.max_open_warning'] = 61

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

main_features = get_feature_df()[['RV', 'IV']]

vix = web.get_data_yahoo('^VIX', const.START_DATE, const.END_DATE)['Adj Close']
vix.name = 'MK_IV'

stock_volume = web.get_data_yahoo(const.STOCK_TICKER, const.START_DATE, const.END_DATE)['Volume']
stock_volume.name = "volume"

pytrend = TrendReq()
pytrend.build_payload(kw_list=['Apple'], timeframe=const.START_DATE.strftime("%Y-%m-%d") + " " +
                                                   const.END_DATE.strftime("%Y-%m-%d"))
trend = pytrend.interest_over_time()['Apple']
trend.name = "trend"

trend = trend.reindex(pd.date_range(const.START_DATE, const.END_DATE, freq='D')).interpolate()
aug_f = vix.to_frame().join(stock_volume).join(trend)

print("volume st test:", adfuller(aug_f['volume']))
print("MK_IV st test:", adfuller(aug_f['MK_IV']))
print("trend st test:", adfuller(aug_f['trend']))

print("realize stationarity")

aug_f['volume'] = aug_f['volume'].pct_change()
aug_f['MK_IV'] = aug_f['MK_IV'].pct_change()
aug_f['trend'] = aug_f['trend'].pct_change()
aug_f = aug_f.dropna()

print("volume st test:", adfuller(aug_f['volume']))
print("MK_IV st test:", adfuller(aug_f['MK_IV']))
print("trend st test:", adfuller(aug_f['trend']))

df = ft_processing.get_target_classification(main_features.join(aug_f))
df[const.TARGET_COLUMN_NAME] = df[const.TARGET_COLUMN_NAME].apply(const.vola_pos_to_int)

autocorr_plot(df)


plt.show()
"""



hot_encoder = OneHotEncoder()


df = ft_processing.stationary(ft_processing.get_target_classification(get_feature_df()).join(vix)
target = df[const.TARGET_COLUMN_NAME].apply(const.vola_pos_to_int)
df = df.iloc[:, 1:]

df = split.train_validation_test(df)[0]
target = (split.train_validation_test(target.to_frame())[0]).iloc[:, 0]

print(df)
print(target)



corr_df = pd.DataFrame(columns=df.columns[1:], index=range(1, 100)).apply(
    lambda x: target.to_frame().join(df.shift(x.name)).corr().iloc[1:, 0], axis='columns')

print(corr_df)
dataframe_plot(df)
print(corr_df.idxmax())
corr_df.plot()
plt.show()

autocorr_plot(df)
autocorr_plot(target.to_frame())"""


