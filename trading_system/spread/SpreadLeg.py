"""

File che implementa lo spread leg

"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

import main_dir.constant as const
from main_dir.constant import OptionType, PositionType


def _option_price_greeks(option_data, option_type, position_type):
    """
    Funzione che calcola il volare dell'opzione nel tempo

    :param option_data: dataframe con i dati del'opzione nel tempo
    :type option_data: pd.DataFrame
    :param option_type: indica il tipo di opzione (tra call e put)
    :type option_type: OptionType
    :param position_type: indica la posizione sull'opzione (tra call e put)
    :type position_type: PositionType

    :return: valore dell'opzione
    :rtype: pd.Series
    """
    option_value = pd.DataFrame(columns=["price", 'delta', 'vega', 'theta'])

    if option_type == OptionType.call:
        option_value['price'] = -option_data['call_price']
        option_value['delta'] = option_data['call_delta']
        option_value['vega'] = option_data['call_vega']
        option_value['theta'] = option_data['call_theta']
    else:
        option_value['price'] = -option_data['put_price']
        option_value['delta'] = option_data['put_delta']
        option_value['vega'] = option_data['put_vega']
        option_value['theta'] = option_data['put_theta']

    if position_type == PositionType.short:
        option_value = -option_value
    return option_value


class SpreadLeg:
    """
    Classe che definisce la singola componente dello spread

    stato interno: _start_date, _option_name, _history, _start_strike
    """

    def __init__(self, option_chain, start_date, option_name, option_type, position_type, quantity=1):
        """
        :param option_chain: dataframe con tutti i dati delle opzioni
        :type option_chain: pd.DataFrame
        :param start_date: data d'inizio
        :type start_date: dt.datetime
        :param option_name: ticker dell'opzione
        :type option_name: str
        :param option_type: indica il tipo di opzione (tra call e put)
        :type option_type: OptionType
        :param position_type: indica la posizione sull'opzione (tra call e put)
        :type position_type: PositionType
        :param quantity: numero di opzioni identiche che compongono la leg
        :type quantity: int
        """
        option_chain_selected = option_chain.loc[option_chain["option"] == option_name]
        option_chain_selected = option_chain_selected.reset_index(level=1).sort_index().drop(columns=['option'])

        self._history = _option_price_greeks(option_chain_selected.loc[start_date:].iloc[:const.TRADING_HOLDING_PERIOD],
                                             option_type, position_type) * quantity
        self._premium = self._history['price'][0]
        self._history.insert(0, column='return', value=self._premium - self._history['price'])  # perché sono negativi

        self._start_date = start_date
        self._option_name = option_name
        self._start_strike = option_chain_selected.loc[start_date, 'strike']
        self._option_type = option_type
        self._position_type = position_type
        self._quantity = quantity

    def get_history(self):
        """Getter di history
        :rtype: pd.DataFrame
        """
        return self._history

    def get_option_name(self):
        """Getter di option_name
        :rtype: str
        """
        return self._option_name

    def get_start_date(self):
        """Getter di start_date
        :rtype: dt.datetime
        """
        return self._start_date

    def get_start_strike(self):
        """Getter di start_strike
        :rtype: float
        """
        return self._start_strike

    def get_premium(self):
        """Getter di premium
        :rtype: float
        """
        return self._premium

    def plot(self, ax_to_plot, start_stock_price, scarto=False):
        """ Funzione che plotta su un asse la linea dello strike

        :param ax_to_plot: asse su cui fare il plot
        :type ax_to_plot: plt.Axes
        :param start_stock_price: prezzo iniziale della stock
        :type start_stock_price: float
        :param scarto: indica se applicare scarto randomico sullo strike (per evitare sovrapposizioni)
        :type scarto: bool
        """
        label, color = None, None  # il colore varia sulla vista bullish o bearish (forte o debole) del mercato

        match (self._position_type, self._option_type):
            case (PositionType.long, OptionType.call):
                label = str(self._quantity) + "x long call " + str(int(np.round(self._start_strike, 0))) + "%"
                color = "green"
            case (PositionType.short, OptionType.call):
                label = str(self._quantity) + "x short call " + str(int(np.round(self._start_strike, 0))) + "%"
                color = "salmon"
            case (PositionType.long, OptionType.put):
                label = str(self._quantity) + "x long put " + str(int(np.round(self._start_strike, 0))) + "%"
                color = "red"
            case (PositionType.short, OptionType.put):
                label = str(self._quantity) + "x short put " + str(int(np.round(self._start_strike, 0))) + "%"
                color = "greenyellow"

        scarto_val = ((1 if np.random.random() < 0.5 else -1) * np.random.randint(25) * 0.009) if scarto else 0
        strike = self._start_strike / 100 * start_stock_price + scarto_val
        ax_to_plot.axhline(y=strike, label=label, color=color)
