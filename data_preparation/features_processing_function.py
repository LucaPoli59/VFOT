"""

Libreria che contiene tutte le funzioni di processing, estrazione, elaborazione delle feature

"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from main_dir import constant as const
from data_preparation import split_functions as split
from data_preparation.ScalerDictionaries.MatchScalers import match_scalers


def _compute_vola_position(vola_spread):
    """
    Funzione d'incapsulamento che calcola a partire dallo spread di volatilità la posizione da prendere

    :param vola_spread: Serie del vola spread (rv - iv)
    :type vola_spread: pd.Series
    :return: serie delle posizione (date, pos)
    :rtype: pd.Series
    """
    vola_df_pos = vola_spread.loc[vola_spread >= 0].to_frame()
    vola_df_neg = vola_spread.loc[vola_spread < 0].to_frame()

    vola_df_pos['threshold'] = _compute_vola_threshold(vola_df_pos['vola_spread'], const.TRADING_STRATEGY_THRESHOLD)
    vola_df_neg['threshold'] = _compute_vola_threshold(vola_df_neg['vola_spread'], 1 - const.TRADING_STRATEGY_THRESHOLD)

    neutral, long, short = const.VolaPosition.neutral, const.VolaPosition.long_vola, const.VolaPosition.short_vola

    long_pos = vola_df_pos.apply(lambda x: neutral if x['vola_spread'] < x['threshold'] else long, axis='columns')
    short_pos = vola_df_neg.apply(
        lambda x: neutral if abs(x['vola_spread']) < x['threshold'] else short, axis='columns')

    return pd.concat([long_pos, short_pos])


def _compute_vola_threshold(vola_spread, quantile_threshold):
    """
    Funzione d'incapsulamento che calcola a partire dallo spread di volatilità la soglia stabilità

    :param vola_spread: Serie del vola spread (rv - iv)
    :type vola_spread: pd.Series
    :param quantile_threshold: parametro che indica il quantile da cui calcolare la threshold
    :type quantile_threshold: float
    :return: serie della soglia (date, threshold)
    :rtype: pd.Series
    """
    threshold = np.abs(vola_spread.rolling(125).quantile(quantile_threshold))
    threshold.loc[threshold.isna()] = threshold.loc[threshold.isna()].rolling(5).mean()
    return threshold


def get_target_classification(feature_df):
    """Funzione che ottiene la variabile target da un dataset di feature

    La variabile generata viene aggiunta come colonna **target** nel dataset, a indice ZERO.

    :param feature_df: dataframe d'input con le variabili indipendenti
    :type feature_df: pd.DataFrame

    :return: dataframe con anche la variabile target
    :rtype: pd.DataFrame
    """

    df_out = feature_df.copy()

    vola_spread = (feature_df[const.MainFeatures.realized_volatility].shift(-const.PREDICTION_STEP) /
                   feature_df[const.MainFeatures.implied_volatility] - 1) * 100

    vola_spread.name = 'vola_spread'
    df_out.insert(0, column=const.TARGET_COLUMN_NAME, value=_compute_vola_position(vola_spread))

    df_out = df_out.dropna()
    return df_out


def get_naive_classification(feature_df):
    """Funzione che ottiene il target naive (ovvero il forecast più stupido)

    :param feature_df: dataframe d'input con le variabili indipendenti
    :type feature_df: pd.DataFrame

    :return: dataframe con anche la variabile target
    :rtype: pd.Series
    """

    # log(^RV / IV)
    # return feature_df[const.MainFeatures.realized_volatility] / feature_df[const.MainFeatures.implied_volatility] # log(^RV / IV)

    vola_spread = (feature_df[const.MainFeatures.realized_volatility] /
                   feature_df[const.MainFeatures.implied_volatility] - 1) * 100

    vola_spread.name = 'vola_spread'
    return _compute_vola_position(vola_spread)


def reverse_target_classification(target_series):
    """Funzione che prende la variabile target e la riconverte nel formato originale

    :param target_series: serie del target
    :type target_series: pd.Series

    :return: serie del target nel formato originale
    :rtype: pd.Series
    """

    # return np.exp(target_series)  # log(^RV / IV)
    return target_series


def stationary(feature_df):
    """Funzione che modifica il feature_df rendendo le feature stabilite stazionarie

    :param feature_df: dataframe d'input con le variabili indipendenti
    :type feature_df: pd.DataFrame

    :return: dataframe con le eventuali feature stazionarie
    :rtype: pd.DataFrame
    """
    feature_df = feature_df.apply(lambda column: column.pct_change() if (
            column.name in const.FEATURE_PCT_CHANGE and const.FEATURE_PCT_CHANGE[column.name]) else column)
    feature_df.dropna(inplace=True)

    return feature_df


def smoothing(feature_df):
    """Funzione che modifica il feature_df applicando lo smoothing alle feature richieste

    :param feature_df: dataframe d'input con le variabili indipendenti
    :type feature_df: pd.DataFrame

    :return: dataframe con le eventuali feature levigate
    :rtype: pd.DataFrame
    """
    feature_df = feature_df.apply(lambda column: savgol_filter(column.values, const.FEATURE_SMOOTHING_WINDOW, 3) if (
            column.name in const.FEATURE_SMOOTHING and const.FEATURE_SMOOTHING[column.name]) else column)
    feature_df.dropna(inplace=True)

    return feature_df


def feature_pca(feature_df):
    """Funzione che trasforma il dataframe delle feature in variabili non correlate

    :param feature_df: dataframe d'input con le variabili indipendenti
    :type feature_df: pd.DataFrame

    :return: dataframe con le feature non correlate
    :rtype: pd.DataFrame
    """
    pca_processor = PCA().fit(split.train_validation_test(feature_df)[0])
    data_final = pca_processor.transform(feature_df)
    return pd.DataFrame(index=feature_df.index, columns=feature_df.columns, data=data_final)


def feature_selection(feature_df, target):
    """Funzione che seleziona tramite lasso regression le feature più importanti

    :param feature_df: dataframe d'input con le variabili indipendenti non scalati
    :type feature_df: pd.DataFrame
    :param target: serie d'input con la variabile dipendente (non scalato con hot encoding)
    :type target: pd.Series

    :return: dataframe con le feature selezionate
    :rtype: pd.DataFrame
    """

    X_data = split.train_validation_test(feature_df)[0]
    y_data = (split.train_validation_test(target.to_frame())[0]).loc[X_data.index, const.TARGET_COLUMN_NAME]

    scalers = match_scalers()
    scalers.add(X_data.columns.tolist())
    scalers.fit(X_data)
    X_data = scalers.transform(X_data)

    logistic_regressor = LogisticRegression(C=0.01, multi_class='multinomial',  penalty='l2')
    logistic_selector = RFE(estimator=logistic_regressor, n_features_to_select=0.5)
    logistic_selector.fit(X_data, y_data)

    return pd.DataFrame(index=feature_df.index,
                        columns=logistic_selector.get_feature_names_out(input_features=feature_df.columns),
                        data=logistic_selector.transform(feature_df))
