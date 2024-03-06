"""

File che definisce la classe positionType, option_type e l'astratta spread

"""

from abc import ABC, abstractmethod
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

from main_dir import constant as const
from main_dir.constant import VolaPosition


def _compute_stock_position(history, hedging_threshold):
    """
    Funzione che calcola edging dello spread aggiungendo la stock

    :param history: history dello spread senza l'hedging
    :type history: pd.DataFrame
    :param hedging_threshold: soglia in cui l'hedging si attiva
    :type hedging_threshold: pd.DataFrame

    :return: history hedged
    :rtype: pd.DataFrame
    """
    history['stock_position_size'] = 0
    data_prev = history.index[0]

    for data in history.iloc[1:].index:
        history.at[data, 'delta'] = history.at[data, 'delta'] + history.at[data_prev, 'stock_position_size']
        if abs(history.at[data, 'delta']) > hedging_threshold:
            history.at[data, 'stock_position_size'] = history.at[data_prev, 'stock_position_size'] - \
                                                      history.at[data, 'delta']
            history.at[data, 'delta'] = 0
        else:
            history.at[data, 'stock_position_size'] = history.at[data_prev, 'stock_position_size']
        data_prev = data

    return history


class Spread(ABC):
    """
    Classe astratta generica dello spread che stabilisce i metodi generici, include anche la "classe" position type

    L'idea della classe è quella di aprire un nuovo spread ogni giorno, di tipo long o short
    con una durata e un ritorno finale, e valori (prezzo e greeks) che variano

    Uno spread è l'aggregazione di più legs
    """

    def __init__(self, option_chain, stock_price, start_date, option_candidates, vola_position, hedging_threshold=0,
                 gaps_factor=1):
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
        :param hedging_threshold: soglia in cui l'hedging si attiva
        :type hedging_threshold: pd.DataFrame
        :param gaps_factor: fattore con cui incrementare o decrementare la distanza fra gli strike
        :type gaps_factor: float
        """

        self._legs = []
        self._start_date = start_date
        self._vola_position = vola_position

        if vola_position == VolaPosition.neutral:  # lo spread è vuoto
            self._premium = 0
            self._history_not_hedged = pd.DataFrame()
            self._history = pd.DataFrame()
        else:
            self._compute_strikes(vola_position, gaps_factor)
            self._legs_creation(option_chain, option_candidates, start_date, vola_position)

            self._premium = sum([leg.get_premium() for leg in self._legs])
            self._history_not_hedged = sum([leg.get_history() for leg in self._legs])

            if hedging_threshold != 0:
                self._hedge_history(stock_price, hedging_threshold)
            else:
                self._history = self._history_not_hedged
                self._history['stock_position_size'] = 0

    @abstractmethod
    def _compute_strikes(self, vola_position, gaps_factor):
        """Funzione che calcola il dizionario degli strike (simile a quello in const; con i layer "interni")

        :param vola_position: indica la posizione sulla volatilità dello spread
        :type vola_position: VolaPosition
        :param gaps_factor: fattore con cui incrementare o decrementare la distanza fra gli strike
        :type gaps_factor: float
        """
        pass

    @abstractmethod
    def _legs_creation(self, option_chain, option_candidates, start_date, vola_position):
        """Funzione che crea le legs dello spread

        :param option_chain: dataframe con tutti i dati delle opzioni
        :type option_chain: pd.DataFrame
        :param option_candidates: dataframe con le opzioni candidate a comporre lo spread
        :type option_candidates: pd.DataFrame
        :param start_date: data d'inizio
        :type start_date: dt.datetime
        """
        pass

    def get_start_date(self):
        """Getter di start_date
        :rtype: dt.datetime
        """
        return self._start_date

    def get_vola_position(self):
        """Getter di vola_position
        :rtype: VolaPosition
        """
        return self._vola_position

    def get_history(self):
        """Getter di history
        :rtype: pd.DataFrame
        """
        return self._history

    def get_premium(self):
        """Getter di premium
        :rtype: float
        """
        return self._premium

    def get_total_return(self):
        if self._history.empty:
            return 0
        else:
            return np.round(self._history['return'][-1] / np.abs(self._premium), 3)

    def _hedge_history(self, stock_price, hedging_threshold):
        """Funzione che effettua il delta hedging modificando l'history dello spread
        :param stock_price: serie dei prezzi della stock
        :type stock_price: pd.Series
        :param hedging_threshold: soglia in cui l'hedging si attiva
        :type hedging_threshold: pd.DataFrame
        """

        self._history_not_hedged['delta[nH]'] = self._history_not_hedged['delta']
        self._history_not_hedged['spread_return'] = self._history_not_hedged['return']
        history_hedged = _compute_stock_position(self._history_not_hedged, hedging_threshold)

        history_hedged['stock_price'] = stock_price.loc[history_hedged.index[0]:history_hedged.index[-1]]
        stock_rtn = stock_price.loc[history_hedged.index[0]:history_hedged.index[-1]].diff()  # ritorno assoluto
        history_hedged['stock_rtn'] = stock_rtn
        history_hedged['stock_position_rtn'] = (stock_rtn *
                                                history_hedged['stock_position_size'].shift()).fillna(0).cumsum()
        history_hedged['return'] = history_hedged['return'] + history_hedged['stock_position_rtn']

        self._history = history_hedged  # .drop(columns=['stock_position_rtn'])

    def plot(self, stock_price, iv, rv, opt_title="", scarto=False):
        """ Funzione che plotta lo spread

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

        stock_price = stock_price.loc[self._start_date:][:const.TRADING_HOLDING_PERIOD]
        rv = rv.loc[self._start_date:][:const.TRADING_HOLDING_PERIOD]
        iv = iv.loc[self._start_date:][:const.TRADING_HOLDING_PERIOD]

        fig, ax = plt.subplots()
        plt.suptitle(opt_title + " Premium: " + str(np.round(self._premium, 2)) +
                     "    ΔVola%: " + str(np.round((rv[-1] - iv[0]) / iv[0] * 100, 2)) +
                     "    IMPL_var%: " + str(np.round(iv[0] / np.sqrt(252 / const.TRADING_HOLDING_PERIOD), 2)) +
                     "    Real_var%: " + str(np.round(rv[-1] / np.sqrt(252 / const.TRADING_HOLDING_PERIOD), 2)),
                     fontsize=12, fontweight='bold', y=0.95)

        ax_stock = plt.subplot2grid((1260, 1), (0, 0), rowspan=600, colspan=1, fig=fig)
        ax_iv_rv = plt.subplot2grid((1260, 1), (620, 0), rowspan=200, colspan=1, fig=fig, sharex=ax_stock)
        ax_delta = plt.subplot2grid((1260, 1), (840, 0), rowspan=200, colspan=1, fig=fig, sharex=ax_stock)
        ax_spread = plt.subplot2grid((1260, 1), (1060, 0), rowspan=200, colspan=1, fig=fig, sharex=ax_stock)

        ax_stock.plot(stock_price, label="Stock Price")

        for leg in self._legs:
            leg.plot(ax_stock, stock_price[0], scarto=scarto)

        ax_stock.legend()

        ax_iv_rv.plot(iv, label="Implied Volatility", color='r')
        ax_iv_rv.plot(rv, label="Realized Volatility", color='g')
        ax_iv_rv.legend()

        if self._history.empty:
            ax_delta.axhline(y=0, label="delta", color='b')
            ax_delta.axhline(y=0, label="stock_pos", color='k')
            ax_spread.axhline(y=0, label="Ritorno spread")
        else:
            ax_delta.plot(self._history['delta'], label="delta", color='b')
            ax_delta.plot(self._history['stock_position_size'], label="stock_pos", color='k')
            ax_spread.plot(self._history['return'], label="Ritorno spread")

        ax_delta.legend()
        ax_spread.legend()
