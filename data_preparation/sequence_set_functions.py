"""

File di libreria secondaria che gestisce i sequence set della libreria xarray

"""

import xarray as xr
import numpy as np
import pandas as pd

from main_dir import constant as const


def generate_sequence_set(dataframe_in, sequence_len=const.SEQUENCE_LEN):
    """Funzione che genera le sequenze temporali a partire da un dataset con uno o più feature

    Genera un xarray.Dataset che contiene le sequenze, ovvero tutte le feature laggate.
    Ha 2 dimensioni (tempo, lag:profondità nella sequenza) e ogni feature è una data variable

    :param dataframe_in: dataframe d'input
    :type dataframe_in: pd.DataFrame
    :param sequence_len: lunghezza della sequenza
    :type sequence_len: int

    :returns: dataset con le feature laggate a sequenza
    :rtype: xr.Dataset
    """

    x_df = xr.Dataset(coords={"time": dataframe_in.index.values[sequence_len - 1:],
                              "lag": np.arange(sequence_len)[::-1]},
                      attrs={"name": "sequence_dataset", "description": "xarray che contiene le sequenze"})

    for column in dataframe_in.columns:
        feature_lag = _lag(dataframe_in[column], sequence_len - 1).dropna().iloc[:, ::-1]
        x_df = x_df.assign({column: (x_df.coords, feature_lag.values)})

    return x_df


def _lag(serie_in, lag_num=const.SEQUENCE_LEN):
    """Funzione che preprocessa i dati

    Effettua l'operazione di lag dei dati, creando lag_num colonne con i valori shiftati,
    dunque per un valore x, il dataset avrà come colonne: x, x-1, x-2, ...., x-lag_num

    :param serie_in: serie d'input da preprocessare
    :type serie_in: pd.Series
    :param lag_num: numero per cui si effettua il lag
    :type lag_num: int

    :return: dataframe con i dati preprocessati
    :rtype: pd.DataFrame
    """

    series_list = [serie_in]

    for i in range(1, lag_num + 1):
        series_list.append(pd.Series(name=str(serie_in.name) + "_lag" + str(i), data=serie_in.shift(i)))

    df_out = pd.concat(series_list, axis='columns').dropna()

    return df_out


def sequence_dataset_reshape_to_array(dataset):
    """Funzione che converte il sequence dataset in un array numpy

        Le dimensioni sono (time_index, sequence_index, feature); ovvero
        (numero di sequenze, giorno nella sequenza, feature)

        :param dataset: dataset da convertire
        :type dataset: xr.Dataset

        :returns: array della shape consona
        :rtype: np.ndarray
        """

    return dataset.to_array().values.swapaxes(0, 2).swapaxes(0, 1)


def get_time_index(dataset):
    """Funzione che prende un dataset e restituisce l'indice temporale

    :param dataset: dataset da cui restituire l'indice di tempo
    :type dataset: xr.Dataset

    :return: time index in formato pandas
    :rtype: pd.Index
    """
    return dataset.coords['time'].to_index()


def sequence_array_converter(dataframe):
    """Funzione che prende un dataframe e lo converte in array di sequenze laggate

    :param dataframe: dataframe X
    :type dataframe: pd.DataFrame

    :return: array di sequenze
    :rtype: np.ndarray
    """
    return sequence_dataset_reshape_to_array(generate_sequence_set(dataframe))