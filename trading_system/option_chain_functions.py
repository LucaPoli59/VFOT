"""

File che calcola dal dataset, contenente tutte le opzioni i dati, da usare per il trading

"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import json

from main_dir import constant as const


def get_option_chain():
    """Funzione che carica da disco i dati delle opzioni con tutte le scadenze

    Il caricamento procede con la raffinazione del dataframe iniziale solo se le impostazioni sono state cambiante,
    altrimenti lo carica direttamente dal file salvato

    :return: dataframe con i dati richiesti, indice=['date', 'strike']
    :rtype: pd.DataFrame
    """

    if _is_to_load():
        df = pd.read_csv(const.LOAD_DATASET_PATH + "option_chain.csv")
        df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")
        df['expiration'] = pd.to_datetime(df['expiration'], format="%Y/%m/%d")
        df = df.set_index(['date', 'strike'], drop=True).sort_index()

        return df

    else:

        option_df = _raw_option_load(const.OPTION_RAW_FILE_PATH)
        option_df = _unification_put_call(option_df)
        option_df.to_csv(const.LOAD_DATASET_PATH + "option_chain.csv")

        return option_df


def _raw_option_load(path):
    """Funzione d'incapsulamento, che carica da disco i dati delle opzioni grezzi, e li raffina parzialmente:
     - creando la colonna 'days_to_expire'
     - trasformando lo strike in percentuale

    :param path: percorso in cui è presente il file
    :type path: str

    :return: dataframe con i dati delle opzioni parzialmente raffinati
    :rtype: pd.DataFrame
    """

    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'], format="%Y/%m/%d")
    data['expiration'] = pd.to_datetime(data['expiration'], format="%Y/%m/%d")
    data = data.set_index('date', drop=True)[['stock_price', 'option symbol', 'mean price', 'expiration', 'strike',
                                              'call/put', 'delta', 'vega', 'theta']]

    data.rename(columns={'option symbol': 'option', 'mean price': 'option_price',
                         'delta': 'option_delta', 'vega': 'option_vega', 'theta': 'option_theta'}, inplace=True)

    data['days_to_expire'] = np.busday_count(data.index.values.astype('M8[D]'),
                                             data['expiration'].values.astype('M8[D]'))

    # lo strike è positivo per strike superiori allo stock, è negativo per strike inferiori alla stock
    data['strike'] = data['strike'] / data['stock_price'] * 100

    # effettuo un primissimo filtraggio dei dati inutili
    data = data.loc[(data['strike'] >= 100 - const.TRADING_DF_RANGE_STRIKE) &
                    (data['strike'] <= 100 + const.TRADING_DF_RANGE_STRIKE)]
    data = data.loc[data['days_to_expire'] <= 255]

    return data.set_index(['strike'], append=True).sort_index().drop(columns=['days_to_expire'])


def _unification_put_call(data):
    """Funzione d'incapsulamento, che unifica sulla stessa riga i dati di put e call uguali

    :param data: dataframe d'input
    :type data: pd.DataFrame

    :return: dataframe con put e call unificate
    :rtype: pd.DataFrame
    """

    data = data.set_index(['expiration'], append=True).sort_index()
    c_data = data[data['call/put'] == 'C'].drop(columns=["call/put", "stock_price"])
    c_data.rename(columns={'option_price': 'call_price', 'option_delta': 'call_delta',
                           'option_vega': 'call_vega', 'option_theta': 'call_theta'}, inplace=True)

    p_data = data[data['call/put'] == 'P'].drop(columns=["call/put", "stock_price", 'option'])
    p_data.rename(columns={'option_price': 'put_price', 'option_delta': 'put_delta',
                           'option_vega': 'put_vega', 'option_theta': 'put_theta'}, inplace=True)

    data = c_data.join(p_data).reset_index(level=2).sort_index()
    return data


def _is_to_load():
    """Funzione che stabilisce a seconda del file di config se scaricare o caricare il dataframe da disco

    :return: indicatore se le configurazioni sono rimaste uguali e quindi i dati non vanno riscaricati
    :rtype: bool
    """
    config_file = open(const.LOAD_MODEL_PATH + "\\config.json")
    config_data = json.load(config_file)
    config_file.close()

    return config_data['option_df']['trading_df_range_strike'] == const.TRADING_DF_RANGE_STRIKE


def select_start_date(option_chain, days_to_expire=const.TRADING_HOLDING_PERIOD):
    """Funzione che seleziona le date con opzioni a scadenza pari a days_to_expire

    :param option_chain: dataframe con le tutte le opzioni
    :type option_chain: pd.DataFrame
    :param days_to_expire: giorni alla scadenza dell'opzione
    :type days_to_expire: int

    :return: dataframe con le opzioni selezionate
    :rtype: pd.DataFrame
    """

    expires_candidate = option_chain.loc[option_chain['expiration'] >= option_chain.index.get_level_values(0)
                                         + BDay(days_to_expire)]

    near_expires = expires_candidate.groupby(level=(0, 1)).first()
    best_expires = near_expires.loc[near_expires['expiration'] == near_expires.groupby(level=0)['expiration'].transform(
        lambda date_group:  np.full(len(date_group), date_group.mode()[0]))]

    return best_expires.loc[best_expires['expiration'] ==
                            best_expires.index.get_level_values(0) + BDay(days_to_expire)].iloc[:-1]


def select_strike(option_df, strike):
    """Funzione che a partire da un df con strike multipli, ottiene un dataframe con lo strike selezionato
    n.b. Si prende lo strike più vicino

     :param option_df: df con strike multipli [indice=strike]
     :type option_df: pd.DataFrame
     :param strike: strike da selezionare
     :type strike: float

     :return: option_name
     :rtype: str
     """
    return option_df.iloc[np.argmin(abs(option_df.index.values - strike))]['option']
