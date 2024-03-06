"""
File che dichiara la classe ScalerDictionaryRM

"""


from sklearn.preprocessing import MinMaxScaler, RobustScaler

from main_dir import constant as const
from data_preparation.ScalerDictionaries.ScalerDictionary import ScalerDictionary


class ScalerDictionaryRM(ScalerDictionary):
    """
    Il suo CustomScaler è del tipo (RobustScaler, MinMaxScaler)
    R = robust scaler
    M = MinMax scaler
    l'ordine indica l'ordine di esecuzione B -> M
    """

    def _create(self):
        return list([RobustScaler(), MinMaxScaler(feature_range=(-1, 1))])

    def _fit_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]

        robust_scaler = scaler[0].fit(feature_series.values.reshape(-1, 1))
        robust_scaled = scaler[0].transform(feature_series.values.reshape(-1, 1))
        minmax_scaler = scaler[1].fit(robust_scaled)
        return robust_scaler, minmax_scaler

    def _transform_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]

        robust_scaled = scaler[0].transform(feature_series.values.reshape(-1, 1))
        return scaler[1].transform(robust_scaled)

    def _reverse_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]

        robust_scaled = scaler[1].inverse_transform(feature_series.values.reshape(-1, 1))
        return scaler[0].inverse_transform(robust_scaled)
