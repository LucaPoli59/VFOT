"""
File che dichiara la classe astratta per la gestione degli scaler
"""

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from main_dir import constant as const
from data_preparation.split_functions import target_compression, target_decompression


class ScalerDictionary(ABC):
    """
    Classe astratta generica del dizionario di Scaler

    Esso è un dizionario che associa a ogni feature uno CustomScaler

    Un CustomScaler è una collezione di parametri (outliers range ecc..) e metodi di scaling (MinMax, Quantile ecc...)
    """

    def __init__(self):
        self._scalers = {const.TARGET_COLUMN_NAME: OneHotEncoder(sparse=False)}

    def add(self, feature_name):
        """Funzione che aggiunge i CustomScaler al dizionario

        :param feature_name: nome o lista di nomi
        :type feature_name: str or list
        """

        if type(feature_name) is not list:
            feature_name = [feature_name]

        for name in feature_name:
            if name not in self._scalers:
                self._scalers[name] = self._create()

    @abstractmethod
    def _create(self):
        """
        Metodo che inizializza il CustomScaler per una singola feature

        :return: lista di scaler che compone l'oggetto
        :rtype: list or OneHotEncoder
        """
        pass

    def add_copy(self, new_feature_name, feature_name_to_copy):
        """Funzione che aggiunge i CustomScaler copiandoli da altri giù presenti

        :param new_feature_name: nome o lista di nomi di feature nuove
        :type new_feature_name: str or list
        :param feature_name_to_copy: nome di feature già inserita
        :type feature_name_to_copy: str
        """

        if type(new_feature_name) is not list:
            new_feature_name = [new_feature_name]

        for name in new_feature_name:
            self._scalers[name] = self._scalers[feature_name_to_copy]

    def get(self, feature_name):
        """Effettua la get

        :param feature_name: nome di una feature
        :type feature_name: str

        :return: lo scaler associato
        :rtype: Any or None
        """

        if feature_name in self._scalers:
            return self._scalers[feature_name]
        else:
            return None

    def set(self, feature_name, other):
        """Effettua la set

        :param feature_name: nome di una feature
        :type feature_name: str
        :param other: altro scaler
        :type feature_name: str
        """

        if feature_name in self._scalers:
            self._scalers[feature_name] = other

    def fit(self, df):
        """Funzione che addestra diversi CustomScaler per ogni feature

        :param df: dataframe d'input con cui addestrare lo Scaler
        :type df: pd.DataFrame
        """

        for column in df.columns:
            if column in self._scalers:
                if type(self._scalers[column]) is OneHotEncoder:
                    self._scalers[column] = self._fit_target(df[column])
                else:
                    self._scalers[column] = self._fit_feature(df[column])

    def transform(self, df):
        """Funzione che trasforma il dataframe con il CustomScaler

        :param df: dataframe da trasformare
        :type df: pd.DataFrame

        :return: il dataframe trasformato
        :rtype: pd.DataFrame
        """
        df_out = df.copy()

        for column in df.columns:
            if column in self._scalers:
                if type(self._scalers[column]) is OneHotEncoder:
                    df_out[column] = self._transform_target(df[column])
                else:
                    df_out[column] = self._transform_feature(df[column])

        return df_out

    def reverse(self, df):
        """Funzione che inverte la trasformazione del dataframe con il CustomScaler

        :param df: dataframe da invertire
        :type df: pd.DataFrame

        :return: il dataframe invertito
        :rtype: pd.DataFrame
        """
        df_out = df.copy()

        for column in df_out.columns:
            if column in self._scalers:
                if type(self._scalers[column]) is OneHotEncoder:
                    df_out[column] = self._reverse_target(df[column])
                else:
                    df_out[column] = self._reverse_feature(df[column])

        return df_out

    def _fit_target(self, target_series):
        """Funzione che il custom scaler sul target

        :param target_series: serie d'input con cui addestrare lo Scaler
        :type target_series: pd.Series
        :return il custom scaler addestrato
        """
        return self._scalers[target_series.name].fit(target_series.values.reshape(-1, 1))

    def _transform_target(self, target_series):
        """Funzione che trasforma il target con il CustomScaler

        :param target_series: serie d'input da trasformare
        :type target_series: pd.Series
        :return: la serie trasformata
        :rtype: pd.Series
        """
        return target_compression(pd.DataFrame(index=target_series.index,
                                               data=self._scalers[target_series.name].transform(
                                                   target_series.values.reshape(-1, 1))))

    def _reverse_target(self, target_series):
        """Funzione che inverte la trasformazione del target con il CustomScaler

        :param target_series: serie da invertire
        :type target_series: pd.Series
        :return: la serie invertita
        :rtype: pd.Series
        """
        return self._scalers[target_series.name].inverse_transform(target_decompression(target_series))

    @abstractmethod
    def _fit_feature(self, feature_series):
        """Funzione che il custom scaler su una feature

        :param feature_series: serie d'input con cui addestrare lo Scaler
        :type feature_series: pd.Series
        :return il custom scaler addestrato
        """
        pass

    @abstractmethod
    def _transform_feature(self, feature_series):
        """Funzione che trasforma una feature con il CustomScaler

        :param feature_series: serie d'input da trasformare
        :type feature_series: pd.Series
        :return: la serie trasformata
        :rtype: pd.Series
        """
        pass

    @abstractmethod
    def _reverse_feature(self, feature_series):
        """Funzione che inverte la trasformazione di una feature con il CustomScaler

        :param feature_series: serie da invertire
        :type feature_series: pd.Series
        :return: la serie invertita
        :rtype: pd.Series
        """
