"""

Libreria che contiene tutte le funzioni di split nei sotto dataframe di train validation e test

"""

import pandas as pd
import xarray as xr

from main_dir import constant as const
import data_preparation.sequence_set_functions as s_set


def dataframe_and_sequence_set(dataframe_in, sequence_len=const.SEQUENCE_LEN):
    """Funzione che genera il sequence set a partire dal dataframe e li restituisce a uguale dimensione

    :param dataframe_in: dataframe d'input da dividere
    :type dataframe_in: pd.DataFrame
    :param sequence_len: lunghezza della sequenza
    :type sequence_len: int


    :returns: dataframe, sequence_set
    :rtype: (pd.DataFrame, xr.Dataset)
    """
    sequence = s_set.generate_sequence_set(dataframe_in, sequence_len=sequence_len)
    dataframe_out = dataframe_in.loc[sequence.coords['time'].values[0]:]

    return dataframe_out, sequence


def target_decompression(target_series):
    """Funzione che decomprime il target a partire dalla notazione di lista in una colonna,
    alla notazione valori in più colonne

    :param target_series: serie del target compressa
    :type target_series: pd.Series
    :return: dataframe target decompresso
    :rtype: pd.DataFrame
    """
    return pd.DataFrame.from_records(target_series.values, index=target_series.index)


def target_compression(target):
    """Funzione che decomprime il target a partire dalla notazione di lista in una colonna,
    alla notazione valori in più colonne

    :param target: dataframe o dataset del target decompressa
    :type target: pd.DataFrame or xr.Dataset
    :return: dataframe serie target compressa
    :rtype: pd.Series
    """

    if type(target) is pd.DataFrame:
        s = target.apply(lambda x: list(x), axis='columns')
        s.name = const.TARGET_COLUMN_NAME
        return s

    else:
        target_s = target.sel({"lag": 0}).reset_coords(["lag"], drop=True).to_dataframe().apply(lambda x: list(x),
                                                                                                 axis='columns')
        target_s.index.name = "Date"
        target_s.name = const.TARGET_COLUMN_NAME
        return target_s


def target_and_xdata(dataframe_in, sequence_len=const.SEQUENCE_LEN):
    """Funzione che divide i dati tra variabili indipendenti(target), e dipendente(x).
    Le variabili dipendenti(x) sono restituite nel formato sequence set

    :param dataframe_in: dataframe d'input da dividere
    :type dataframe_in: pd.DataFrame
    :param sequence_len: lunghezza della sequenza
    :type sequence_len: int

    :returns: x_sequence_set, y_sequence_set
    :rtype: (xr.Dataset, xr.Dataset)
    """

    x_sequence_set = s_set.generate_sequence_set(dataframe_in.drop(columns=const.TARGET_COLUMN_NAME),
                                                 sequence_len=sequence_len)
    y_sequence_set = s_set.generate_sequence_set(
        target_decompression(dataframe_in[const.TARGET_COLUMN_NAME]), sequence_len=sequence_len)

    return x_sequence_set, y_sequence_set


def train_validation_test(data_in, train_size=const.TRAIN_SIZE, validation_size=const.VALIDATION_SIZE):
    """Funzione che divide i dati tra train validation e test.

    :param data_in: dataframe o sequence_set d'input da dividere
    :type data_in: pd.DataFrame or xr.Dataset
    :param train_size: dimensione in percentuale del dataframe di train
    :type train_size: float
    :param validation_size: dimensione in percentuale del dataframe di validation
    :type validation_size: float

    nota: le dimensione del test set sono dedotte

    :returns: train, validation e test
    :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame) or (xr.Dataset, xr.Dataset, xr.Dataset)
    """

    if type(data_in) is pd.DataFrame:
        length = len(data_in)
        train_dataframe = data_in.iloc[:int(length * train_size)]
        validation_dataframe = data_in.iloc[int(length * train_size): int(length * (train_size + validation_size))]
        test_dataframe = data_in.iloc[int(length * (train_size + validation_size)):]

        return train_dataframe, validation_dataframe, test_dataframe
    else:
        length = data_in.dims['time']
        train_set = data_in.isel(time=slice(0, int(length * train_size)))
        validation_set = data_in.isel(time=slice(int(length * train_size),
                                                 int(length * (train_size + validation_size))))
        test_set = data_in.isel(time=slice(int(length * (train_size + validation_size)), None))

        return train_set, validation_set, test_set


def train_test(data_in, train_size=const.TRAIN_SIZE):
    """Funzione che divide i dati tra train e test.

    :param data_in: dataframe o sequence_set d'input da dividere
    :type data_in: pd.DataFrame or xr.Dataset
    :param train_size: dimensione in percentuale dei dataset di train
    :type train_size: float

    nota: le dimensione del test set sono dedotte

    :returns: train, test
    :rtype: (pd.DataFrame, pd.DataFrame) or (xr.Dataset, xr.Dataset)
    """

    if type(data_in) is pd.DataFrame:
        train_length = int(len(data_in) * train_size)
        train_dataframe = data_in.iloc[:train_length]
        test_dataframe = data_in.iloc[train_length:]

        return train_dataframe, test_dataframe
    else:
        train_length = int(data_in.dims['time'] * train_size)
        train_set = data_in.isel(time=slice(0, train_length))
        test_set = data_in.isel(time=slice(train_length, None))

        return train_set, test_set


def train_and_validation_test(data_in, train_validation_size=const.TRAIN_VALIDATION_SIZE):
    """Funzione che divide i dati tra train-validation set e test set, tipicamente usata per il grid_search.

    :param data_in: dataframe o sequence_set d'input da dividere
    :type data_in: pd.DataFrame or xr.Dataset
    :param train_validation_size: dimensione in percentuale dei dataset di train-validation
    :type train_validation_size: float

    nota: le dimensione del test set sono dedotte

    :returns: train_validation, test,
    :rtype: (pd.DataFrame, pd.DataFrame) or (xr.Dataset, xr.Dataset)
    """

    if type(data_in) is pd.DataFrame:
        train_validation_length = int(len(data_in) * train_validation_size)
        train_validation_dataframe = data_in.iloc[:train_validation_length]
        test_dataframe = data_in.iloc[train_validation_length:]

        return train_validation_dataframe, test_dataframe
    else:
        train_validation_length = int(data_in.dims['time'] * train_validation_size)
        train_validation_set = data_in.isel(time=slice(0, train_validation_length))
        test_set = data_in.isel(time=slice(train_validation_length, None))

        return train_validation_set, test_set
