"""

Libreria che contiene tutte le funzioni di collection dei dati

"""
import pandas as pd
import pandas_datareader as web
from pytrends.request import TrendReq
from pandas.tseries.offsets import BDay
import numpy as np
import datetime as dt
import requests
import json
import csv


import main_dir.constant as const


def get_feature_df():
    """Funzione che scarica o carica da disco i dati scelti nel progetto

    :return: dataframe con i dati richiesti
    :rtype: pd.DataFrame
    """

    if _is_to_load():
        df = pd.read_csv(const.LOAD_DATASET_PATH + "feature_dataframe.csv")
        df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
        df = df.set_index('Date', drop=True)
        print("feature_dataframe loaded from disk")

    else:
        print("feature_dataframe downloaded")

        df = pd.DataFrame(index=get_price().index, columns=const.FEATURE_LIST)
        df[const.FEATURE_LIST[0]] = get_price()
        df[const.FEATURE_LIST[1]] = _get_rv()
        df[const.FEATURE_LIST[2]] = _get_iv()
        df[const.FEATURE_LIST[3]] = _get_rv_long_term()
        df[const.FEATURE_LIST[4]] = _get_volume()
        df[const.FEATURE_LIST[5]] = _get_pe()
        df[const.FEATURE_LIST[6]] = _get_trend()
        df[const.FEATURE_LIST[7]] = _get_dte()
        df[const.FEATURE_LIST[8]] = _get_dtd()
        df[const.FEATURE_LIST[9]] = _get_hml()
        df[const.FEATURE_LIST[10]] = _get_smb()
        df[const.FEATURE_LIST[11]] = _get_mkr()
        df[const.FEATURE_LIST[12]] = _get_mk_iv()
        df[const.FEATURE_LIST[13]] = _get_rfr()
        df[const.FEATURE_LIST[14]] = _get_infl()
        df.dropna(inplace=True)
        df.to_csv(const.LOAD_DATASET_PATH + "feature_dataframe.csv")

    return df


def _is_to_load():
    """Funzione che stabilisce a seconda del file di config se scaricare o caricare il dataframe da disco

    :return: indicatore se le configurazioni sono rimaste uguali e quindi i dati non vanno riscaricati
    :rtype: bool
    """
    config_file = open(const.LOAD_MODEL_PATH + "\\config.json")
    config_data = json.load(config_file)
    config_file.close()

    macros = config_data['feature_df']

    start_date = dt.datetime.strptime(macros['start'], '%H.%M_%d-%m-%y')
    end_date = dt.datetime.strptime(macros['end'], '%H.%M_%d-%m-%y')

    return macros['feature_list'] == const.FEATURE_LIST and start_date == const.START_DATE \
           and end_date == const.END_DATE and macros['feature_compute_window'] == const.FEATURE_COMPUTE_WINDOW


def get_price(back_days_shift=0):
    """Funzione che scarica il prezzo

    :param back_days_shift: indica quanti giorni precedenti scaricare aggiuntivamente
    :type back_days_shift: int

    :return: Serie del Stock_Price
    :rtype: pd.Series
    """
    start_date = _datetime_b_difference(const.START_DATE, back_days_shift)
    serie = web.get_data_yahoo(const.STOCK_TICKER, start_date, const.END_DATE)['Adj Close']
    serie.name = const.FEATURE_LIST[0]
    return serie


def _get_rv():
    """Funzione che scarica la realized volatility

    :return: Serie della RV
    :rtype: pd.Series
    """

    rtn = np.log1p(get_price(back_days_shift=const.FEATURE_COMPUTE_WINDOW).pct_change().dropna())
    var = rtn.rolling(const.FEATURE_COMPUTE_WINDOW, min_periods=const.FEATURE_COMPUTE_WINDOW).var()
    rv = np.sqrt(var * 252) * 100

    rv.name = const.FEATURE_LIST[1]
    return rv


def _get_iv():
    """Funzione che scarica l'implied volatility

    :return: Serie della IV
    :rtype: pd.Series
    """

    iv = pd.read_csv(const.IV_FILE_PATH)
    iv['date'] = pd.to_datetime(iv['date'], format="%Y/%m/%d")
    iv = iv.set_index('date', drop=True)
    iv = iv['volatility']
    iv.name = const.FEATURE_LIST[2]

    return iv


def _get_rv_long_term():
    """Funzione che scarica la realized volatility long term

    :return: Serie della RV
    :rtype: pd.Series
    """

    rtn = np.log1p(get_price(back_days_shift=252).pct_change().dropna())
    var = rtn.rolling(252, min_periods=252).var()
    rv = np.sqrt(var * 252) * 100

    rv.name = const.FEATURE_LIST[3]
    return rv


def _get_volume():
    """Funzione che scarica volumi giornalieri

    :return: Serie del volume
    :rtype: pd.Series
    """

    stock_volume = web.get_data_yahoo(const.STOCK_TICKER, const.START_DATE, const.END_DATE)['Volume']
    stock_volume.name = const.FEATURE_LIST[4]
    return stock_volume


def _get_pe():
    """Funzione che scarica il price to earnings

    :return: Serie del PE
    :rtype: pd.Series
    """

    eps_full_annualized = _get_eps().rolling(4, min_periods=4).sum().dropna()  # 4 sono i report trimestrali in un anno
    price = get_price()

    eps_annualized = eps_full_annualized.reindex(index=price.index, method="backfill")[const.START_DATE:const.END_DATE]

    pe = price / eps_annualized
    pe.name = const.FEATURE_LIST[5]
    return pe


def _get_trend():
    """Funzione che scarica il trend per l'azione

    :return: Serie del trend
    :rtype: pd.Series
    """

    pytrend = TrendReq()
    pytrend.build_payload(kw_list=['Apple'], timeframe=const.START_DATE.strftime("%Y-%m-%d") + " " +
                                                       const.END_DATE.strftime("%Y-%m-%d"))
    trend = pytrend.interest_over_time()['Apple']
    trend.name = const.FEATURE_LIST[6]
    return trend.reindex(pd.date_range(const.START_DATE, const.END_DATE, freq='D')).interpolate()


def _get_dte():
    """Funzione che scarica i Days to Earnings

    :return: Serie del DtE
    :rtype: pd.Series
    """
    eps_date = _get_eps().index
    eps_series = pd.Series(data=eps_date, index=eps_date)

    full_index_date = get_price().index
    next_eps = eps_series.reindex(index=full_index_date, method='backfill')

    dte = pd.Series(index=next_eps.index, data=np.busday_count(next_eps.index.values.astype('M8[D]'),
                                                               next_eps.values.astype('M8[D]')))
    dte.name = const.FEATURE_LIST[7]
    return dte


def _get_dtd():
    """Funzione che scarica i Days to Dividend

    :return: Serie del DtD
    :rtype: pd.Series
    """
    actions = web.get_data_yahoo_actions("AAPL", const.START_DATE)
    dividend_date = actions.where(actions['action'] == "DIVIDEND").dropna().index[::-1]
    dividend_date = pd.Series(index=dividend_date, data=dividend_date, name="Dividend Date")
    next_div = dividend_date.iloc[-4] + dt.timedelta(days=365)
    dividend_date.loc[next_div] = next_div

    full_index_date = get_price().index
    next_dividend = dividend_date.reindex(index=full_index_date, method='backfill')

    dtd = pd.Series(index=next_dividend.index, data=np.busday_count(next_dividend.index.values.astype('M8[D]'),
                                                                    next_dividend.values.astype('M8[D]')))
    threshold = dtd.describe()["75%"]
    dtd.where(dtd < threshold, threshold, inplace=True)
    dtd.where(dtd > 0, threshold, inplace=True)
    dtd.name = const.FEATURE_LIST[8]
    return dtd


def _get_hml():
    """Funzione che scarica i dati high minus low

    :return: Serie del HML
    :rtype: pd.Series
    """

    df_fama_french = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                                    start=const.START_DATE, end=const.END_DATE)[0]
    hml = df_fama_french['HML'].div(100)
    hml.name = const.FEATURE_LIST[9]
    return hml


def _get_smb():
    """Funzione che scarica i dati small minus large

    :return: Serie del SMB
    :rtype: pd.Series
    """

    df_fama_french = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                                    start=const.START_DATE, end=const.END_DATE)[0]
    smb = df_fama_french['SMB'].div(100)
    smb.name = const.FEATURE_LIST[10]
    return smb


def _get_mkr():
    """Funzione che scarica il Ritorno medio annualizzato del mercato

    :return: Serie del MKR
    :rtype: pd.Series
    """
    returns = get_price(back_days_shift=const.FEATURE_COMPUTE_WINDOW).pct_change().dropna()

    mkr = returns.rolling(const.FEATURE_COMPUTE_WINDOW, min_periods=const.FEATURE_COMPUTE_WINDOW).mean() * 252
    mkr.name = const.FEATURE_LIST[11]

    return mkr


def _get_mk_iv():
    """Funzione che scarica il vix

    :return: Serie del VIX
    :rtype: pd.Series
    """
    vix = web.get_data_yahoo('^VIX', const.START_DATE, const.END_DATE)['Adj Close']
    vix.name = const.FEATURE_LIST[12]

    return vix


def _get_rfr():
    """Funzione che scarica il Ritorno dello strumento risk-free

    :return: Serie del MKR
    :rtype: pd.Series
    """

    rfr = web.get_data_fred("DGS1", const.START_DATE - dt.timedelta(days=3), const.END_DATE).div(100)
    rfr = pd.Series(index=rfr.index.values, data=rfr['DGS1'].values, name=const.FEATURE_LIST[13])
    return rfr.interpolate()


def _get_infl():
    """Funzione che scarica il rateo d'inflazione

    :return: Serie del infl
    :rtype: pd.Series
    """

    infl = web.get_data_fred("T10YIE", const.START_DATE - dt.timedelta(days=3), const.END_DATE)
    infl = pd.Series(data=infl['T10YIE'].values, index=infl.index.values, name=const.FEATURE_LIST[14])
    return infl.interpolate()


def _get_eps():
    """Funzione che scarica gli eps

    :return: Serie degli earnings
    :rtype: pd.Series
    """

    url = "https://www.alphavantage.co/query?function=EARNINGS&symbol=" + const.STOCK_TICKER + \
          "&apikey=" + const.ALPHA_VANTAGE_API
    data = requests.get(url).json()
    tmp_df = pd.DataFrame.from_records(data['quarterlyEarnings'])

    eps = pd.Series(data=tmp_df['reportedEPS'].astype('float64').values, name="EPS",
                    index=pd.DatetimeIndex(data=tmp_df['reportedDate'].values))[::-1]

    url = "https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol=" + const.STOCK_TICKER + \
          "&horizon=3month&apikey=" + const.ALPHA_VANTAGE_API

    last_row = list(csv.reader(requests.get(url).content.decode('utf-8').splitlines(), delimiter=','))[1]

    eps.loc[last_row[2]] = last_row[4]
    return eps


def _datetime_b_difference(date, n_day):
    """Funzione che calcola la differenza in giorni lavorativi tra una data e il numero di giorni, restituendo la data

    *funz d'incapsulamento*

    :param date: data da cui sottrarre
    :type date: dt.datetime
    :param n_day: numero di giorni da sottrarre
    :type n_day: int

    :return: data risultato
    :rtype: dt.datetime
    """

    difference_date = dt.date.fromordinal((date - BDay(n_day)).toordinal())
    return dt.datetime.fromordinal(difference_date.toordinal())
