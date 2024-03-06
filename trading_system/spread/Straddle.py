"""

File che definisce la sotto-classe spread di tipo straddle

"""
import pandas as pd
import datetime as dt

from trading_system.spread.Spread import Spread
from trading_system.spread.SpreadLeg import SpreadLeg
from trading_system.option_chain_functions import select_strike
from main_dir.constant import VolaPosition, OptionType, PositionType
import main_dir.constant as const


class Straddle(Spread):

    def __init__(self, option_chain, stock_price, start_date, option_candidates, vola_position):
        """
        :param option_chain: dataframe con tutti i dati delle opzioni
        :type option_chain: pd.DataFrame
        :param stock_price: serie dei prezzi della stock
        :type stock_price: pd.Series
        :param start_date: data d'inizio
        :type start_date: dt.datetime
        :param option_candidates: dataframe con le opzioni candidate a comporre lo spread
        :type option_candidates: pd.DataFrame
        :param vola_position: indica la posizione sulla volatilità dello spread
        :type vola_position: VolaPosition
       """
        super().__init__(option_chain, stock_price, start_date, option_candidates, vola_position,
                         hedging_threshold=const.TRADING_HEDGING_SPREAD['straddle'])

    def _compute_strikes(self, vola_position, gaps_factor):
        self._strike = const.TRADING_STRIKE_STRADDLE['main']

    def _legs_creation(self, option_chain, option_candidates, start_date, vola_position):

        option_name = select_strike(option_candidates, self._strike)
        if vola_position == VolaPosition.long_vola:
            self._legs.append(SpreadLeg(option_chain, start_date, option_name, OptionType.call, PositionType.long))
            self._legs.append(SpreadLeg(option_chain, start_date, option_name, OptionType.put, PositionType.long))
        else:
            self._legs.append(SpreadLeg(option_chain, start_date, option_name, OptionType.call, PositionType.short))
            self._legs.append(SpreadLeg(option_chain, start_date, option_name, OptionType.put, PositionType.short))

    def plot(self, stock_price, iv, rv, opt_title="", scarto=True):
        """ Funzione che plotta la straddle

       :param stock_price: serie dei prezzi della stock
       :type stock_price: pd.Series
       :param iv: serie dell'implied volatility della stock
       :type iv: pd.Series
       :param rv: serie della realized volatility della stock
       :type rv: pd.Series
       :param opt_title titolo aggiuntivo opzionale
       :type opt_title: str
       :param scarto: indica se applicare scarto randomico sugli strike (per evitare sovrastampe)
        :type scarto: bool
       """
        name = "Long Straddle:" if self._vola_position == VolaPosition.long_vola else "Short Straddle:"
        super().plot(stock_price, iv, rv, opt_title + name, scarto=scarto)
