"""
File che dichiara la classe ScalerDictionaryR
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler

from main_dir import constant as const
from data_preparation.ScalerDictionaries.ScalerDictionary import ScalerDictionary


class ScalerDictionaryR(ScalerDictionary):
    """
    Il suo CustomScaler è del tipo (RobustScaler)
    R = RobustScaler
    l'ordine indica l'ordine di esecuzione R
    """
    def _create(self):
        return RobustScaler()

    def _fit_feature(self, feature_series):
        return self._scalers[feature_series.name].fit(feature_series.values.reshape(-1, 1))

    def _transform_feature(self, feature_series):
        return self._scalers[feature_series.name].transform(feature_series.values.reshape(-1, 1))

    def _reverse_feature(self, feature_series):
        return self._scalers[feature_series.name].inverse_transform(feature_series.values.reshape(-1, 1))