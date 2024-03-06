from data_preparation.ScalerDictionaries.MatchScalers import match_scalers
from data_preparation.data_collection_functions import get_feature_df, get_price
from data_preparation import features_processing_function as ft_processing
from main_dir import output_function as out
import trading_system.option_chain_functions as option_preparation
from data_preparation import split_functions as split
from main_dir import constant as const

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

def corr_plot(dataframe, title):
    corr_matrix = dataframe.corr()
    fig, ax = plt.subplots()
    plt.title(title)
    sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                annot=True, cmap='coolwarm', ax=ax)
    return fig, ax


"""price = get_price()
smoothed = ft_processing.smoothing(price.to_frame()).iloc[:, 0]
smoothed.name = "Serie dei prezzi levigata"
price.name = "Serie dei prezzi"

out.simple_plot(price.to_frame().join(smoothed).dropna().loc["08-01-2020":"04-01-2021"], "Serie dei prezzi a confronto")"""

#out.simple_plot(smoothed.loc["01-01-2020":"01-01-2022"], "Serie dei prezzi smoothed")


df = ft_processing.stationary(get_feature_df())
scalers = match_scalers()
scalers.add(df.columns.tolist())
scalers.fit(split.train_validation_test(df)[0])
df_scaled = scalers.transform(df)



df_pca = ft_processing.feature_pca(df)
df_scaled_pca = ft_processing.feature_pca(df_scaled)

scalers = match_scalers()
scalers.add(df_scaled_pca.columns.tolist())
scalers.fit(split.train_validation_test(df_scaled_pca)[0])
df_scaled_pca_scaled = scalers.transform(df_scaled_pca)

out.simple_plot(df['Stock_Price'].to_frame(), 'stazionario')
corr_plot(df, 'stazionario')
out.simple_plot(df_scaled['Stock_Price'].to_frame(), 'scalato')
corr_plot(df_scaled, 'scalato')

out.simple_plot(df_pca['Stock_Price'].to_frame(), 'stazionario pca')
corr_plot(df_pca, 'stazionario pca')
out.simple_plot(df_scaled_pca['Stock_Price'].to_frame(), 'scalato pca')
corr_plot(df_scaled_pca, 'scalato pca')
out.simple_plot(df_scaled_pca_scaled['Stock_Price'].to_frame(), 'scalato pca e riscalato')
corr_plot(df_scaled_pca, 'scalato pca e riscalato ')


out.show_all_plot()
