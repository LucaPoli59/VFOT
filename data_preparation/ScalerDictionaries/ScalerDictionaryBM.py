"""
File che dichiara la classe ScalerDictionaryBM

"""


from sklearn.preprocessing import MinMaxScaler

from main_dir import constant as const
from data_preparation.ScalerDictionaries.ScalerDictionary import ScalerDictionary


class ScalerDictionaryBM(ScalerDictionary):
    """
    Il suo CustomScaler è del tipo (MinMax_scaler, lower_bound, upper_bound)
    B = bounds (upper e lower)
    M = MinMax scaler
    l'ordine indica l'ordine di esecuzione B -> M
    """
    def _create(self):
        return list([MinMaxScaler(feature_range=(-1, 1)), None, None])

    def _fit_feature(self, feature_series):
        stats = feature_series.describe(percentiles=[const.OUTLIERS_THRESHOLD,
                                                     1 - const.OUTLIERS_THRESHOLD])
        q1 = stats[str(int(const.OUTLIERS_THRESHOLD * 100)) + "%"]
        q3 = stats[str(int((1 - const.OUTLIERS_THRESHOLD) * 100)) + "%"]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        minmax_scaler = self._scalers[feature_series.name][0].fit(feature_series.values.reshape(-1, 1))
        return minmax_scaler, lower_bound, upper_bound

    def _transform_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]
        feature_series.where(feature_series > scaler[1], scaler[1], inplace=True)
        feature_series.where(feature_series < scaler[2], scaler[2], inplace=True)
        return scaler[0].transform(feature_series.values.reshape(-1, 1))

    def _reverse_feature(self, feature_series):
        return self._scalers[feature_series.name][0].inverse_transform(feature_series.values.reshape(-1, 1))
