"""

File che definisce la sotto-classe spread di tipo Butterfly

"""
import pandas as pd
import datetime as dt

from trading_system.spread.Spread import Spread
from trading_system.spread.SpreadLeg import SpreadLeg
from trading_system.option_chain_functions import select_strike
from main_dir.constant import VolaPosition, OptionType, PositionType
import main_dir.constant as const


class Butterfly(Spread):

    def __init__(self, option_chain, stock_price, start_date, option_candidates, vola_position, gaps_factor=1):
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
        :param gaps_factor: fattore con cui incrementare o decrementare la distanza fra gli strike
        :type gaps_factor: float
       """
        self._hedging = const.TRADING_HEDGING_SPREAD['butterfly']
        super().__init__(option_chain, stock_price, start_date, option_candidates, vola_position,
                         gaps_factor=gaps_factor, hedging_threshold=const.TRADING_HEDGING_SPREAD['butterfly'])

    def _compute_strikes(self, vola_position, gaps_factor):
        main = const.TRADING_STRIKE_BUTTERFLY['main']
        left_strike = const.TRADING_STRIKE_BUTTERFLY['coverage']['left'][vola_position] * gaps_factor
        right_strike = const.TRADING_STRIKE_BUTTERFLY['coverage']['right'][vola_position] * gaps_factor

        self._strikes = {'main': main, 'coverage':  {'left': left_strike, 'right': right_strike}}

    def _legs_creation(self, option_chain, option_candidates, start_date, vola_position):
        main_strike = self._strikes['main']
        left_strike, right_strike = self._strikes['coverage'].values()

        main_option = select_strike(option_candidates, main_strike)
        left_option = select_strike(option_candidates, left_strike)
        right_option = select_strike(option_candidates, right_strike)

        if vola_position == VolaPosition.long_vola:
            self._legs.append(SpreadLeg(option_chain, start_date, left_option, OptionType.call, PositionType.short))
            self._legs.append(SpreadLeg(option_chain, start_date, main_option, OptionType.call, PositionType.long,
                                        quantity=2))
            self._legs.append(SpreadLeg(option_chain, start_date, right_option, OptionType.call, PositionType.short))
        else:
            self._legs.append(SpreadLeg(option_chain, start_date, left_option, OptionType.call, PositionType.long))
            self._legs.append(SpreadLeg(option_chain, start_date, main_option, OptionType.call, PositionType.short,
                                        quantity=2))
            self._legs.append(SpreadLeg(option_chain, start_date, right_option, OptionType.call, PositionType.long))

    def plot(self, stock_price, iv, rv, opt_title="", scarto=False):
        """ Funzione che plotta la butterfly

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
        name = "Short Butterfly:" if self._vola_position == VolaPosition.long_vola else "Long Butterfly:"
        super().plot(stock_price, iv, rv, opt_title + name, scarto=scarto)
