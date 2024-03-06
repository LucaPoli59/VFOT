"""
File che dichiara la classe ScalerDictionaryQBM
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from main_dir import constant as const
from data_preparation.ScalerDictionaries.ScalerDictionary import ScalerDictionary


class ScalerDictionaryQBM(ScalerDictionary):
    """
    Il suo CustomScaler è del tipo (QuantileTransformer, MinMax_scaler, lower_bound, upper_bound)
    Q = Quantile transformer
    B = bounds (upper e lower)
    M = MinMax scaler
    l'ordine indica l'ordine di esecuzione Q -> B -> M
    """

    def _create(self):
        return list([QuantileTransformer(output_distribution='normal'),
                     MinMaxScaler(feature_range=(-1, 1)), None, None])

    def _fit_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]

        quantile_transformer = scaler[0].fit(feature_series.values.reshape(-1, 1))
        q_transformed = pd.Series(index=feature_series.index, data=quantile_transformer.transform(
            feature_series.values.reshape(-1, 1)).reshape(-1))

        stats = q_transformed.describe(percentiles=[const.OUTLIERS_THRESHOLD, 1 - const.OUTLIERS_THRESHOLD])
        q1 = stats[str(int(const.OUTLIERS_THRESHOLD * 100)) + "%"]
        q3 = stats[str(int((1 - const.OUTLIERS_THRESHOLD) * 100)) + "%"]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        q_transformed.where(q_transformed > lower_bound, lower_bound, inplace=True)
        q_transformed.where(q_transformed < upper_bound, upper_bound, inplace=True)
        minmax_scaler = scaler[1].fit(q_transformed.values.reshape(-1, 1))
        return quantile_transformer, minmax_scaler, lower_bound, upper_bound

    def _transform_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]

        q_transformed = pd.Series(index=feature_series.index, data=scaler[0].transform(
            feature_series.values.reshape(-1, 1)).reshape(-1))

        q_transformed.where(q_transformed > scaler[2], scaler[2], inplace=True)
        q_transformed.where(q_transformed < scaler[3], scaler[3], inplace=True)
        return scaler[1].transform(q_transformed.values.reshape(-1, 1))

    def _reverse_feature(self, feature_series):
        scaler = self._scalers[feature_series.name]

        min_max_t = scaler[1].inverse_transform(feature_series.values.reshape(-1, 1))
        return scaler[0].inverse_transform(min_max_t)