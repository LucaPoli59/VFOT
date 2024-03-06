"""
File che dichiara la classe ScalerDictionaryR
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from main_dir import constant as const
from data_preparation.ScalerDictionaries.ScalerDictionary import ScalerDictionary


class ScalerDictionaryM(ScalerDictionary):
    """
    Il suo CustomScaler è del tipo (MinMaxScaler)
    M = MinMaxScaler
    l'ordine indica l'ordine di esecuzione M
    """

    def _create(self):
        return MinMaxScaler(feature_range=(-1, 1))

    def _fit_feature(self, feature_series):
        return self._scalers[feature_series.name].fit(feature_series.values.reshape(-1, 1))

    def _transform_feature(self, feature_series):
        return self._scalers[feature_series.name].transform(feature_series.values.reshape(-1, 1))

    def _reverse_feature(self, feature_series):
        return self._scalers[feature_series.name].inverse_transform(feature_series.values.reshape(-1, 1))
