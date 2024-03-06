"""

File che include funzioni e classi per il trading

"""

import pandas as pd
import numpy as np

from main_dir import constant as const
from trading_system.spread.Straddle import Straddle
from trading_system.spread.Strangle import Strangle
from trading_system.spread.Butterfly import Butterfly
from trading_system.spread.IronCondor import IronCondor
from trading_system.option_chain_functions import select_start_date


def apply_trading_strategy(iv, rv):
    """
    Funzione stabilisce (come nella teoria della strategia) quando andare long o short sulla volatilità

    :param iv: iv della stock (allineato con l'option chain)
    :type iv: pd.Series
    :param rv: rv della stock (allineato con l'option chain)
    :type rv: pd.Series
    :return: strategy_position: le posizioni prese dalla strategia
    :rtype pd.Series
    """
    rv_forward = rv.shift(-const.TRADING_HOLDING_PERIOD)
    vola_df = (rv_forward - iv) / iv * 100
    vola_df.name = 'delta_vola'
    vola_df = vola_df.to_frame()

    vola_df['threshold'] = np.abs(vola_df['delta_vola']).groupby(vola_df['delta_vola'] < 0)\
        .rolling(125, min_periods=125).quantile(const.TRADING_STRATEGY_THRESHOLD)\
        .reset_index(level=0, drop=True).sort_index()

    return vola_df.apply(lambda x: _assign_position(x['delta_vola'], x['threshold']), axis='columns')


def _assign_position(delta_vola, threshold):

    return const.VolaPosition.neutral if abs(delta_vola) < threshold else \
        const.VolaPosition.short_vola if delta_vola < 0 else const.VolaPosition.long_vola


def generate_spreads(option_chain, stock_price, strategy_position):
    """Funzione che crea gli spread a partire dalle opzioni presenti in option chain con le posizioni specificate
    in strategy_position

     :param option_chain: dataframe con tutte le opzioni
     :type option_chain: pd.DataFrame
     :param stock_price: serie dei prezzi della stock
     :type stock_price: pd.Series
     :param strategy_position: serie che per ogni data indica se andare short o long sulla volatilità
     :type strategy_position: pd.Series

     :return: spread_dataframe: dataframe contenente gli spread
     :rtype: pd.DataFrame
     """
    candidates = select_start_date(option_chain)

    return pd.DataFrame(index=candidates.index.get_level_values(0).unique(),
                        columns=['straddle', 'strangle', 'butterfly', 'iron_condor'],
                        data=
                        [(Straddle(option_chain, stock_price, date, candidates.loc[date], strategy_position[date]),
                          Strangle(option_chain, stock_price, date, candidates.loc[date], strategy_position[date]),
                          Butterfly(option_chain, stock_price, date, candidates.loc[date], strategy_position[date]),
                          IronCondor(option_chain, stock_price, date, candidates.loc[date], strategy_position[date]))
                         for date in candidates.index.get_level_values(0).unique()])
