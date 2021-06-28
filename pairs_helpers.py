"""Helper Functions to Support Pairs Trading

This file can be imported as a module and contains the following functions:
    * create_and_save_historicals - returns a df with all coin information
    * binance_data_to_df - historical information for a single coin
    * two_coin_pricing - historical log pricing of two coins and their difference in log pricing
    * single_stationarity_test - stationarity test for a pd series
    * pair_stationarity_test - stationarity test for the difference in log pricing between two coins
    * potential_pairs - list of all coin pairs that are stationary and have sufficient trade volume
    * ranked_crossing - ranked list of coin pairs based on how often they mean reverted
    * ranked_volatility - ranked list of coin pairs based on volatility
    * composite_ranking - ranked list of coin pairs combining ranked_crossing and ranked_volatility
"""

import pandas as pd
import numpy as np
import math
import os
import os.path
from datetime import datetime, timedelta, datetime
from dateutil import parser
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations
import binance
from binance.client import Client
from typing import Union, Optional


BINANCE_CLIENT = Client(os.getenv('BINANCE_KEY'), os.getenv('BINANCE_SECRET_KEY'))

BINSIZES = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
# batch_size = 750
BINANCE_SYMBOLS = ['1INCHBTC', 'AAVEBTC', 'ADABTC', 'ALGOBTC', 'ALICEBTC', 'ALPHABTC', 'AMBBTC', 'ANKRBTC', 'ASTBTC', 'ATOMBTC', 'AVAXBTC', 'BATBTC', 'BCHBTC', 'BLZBTC', 'BNBBTC', 'BQXBTC', 'CELRBTC', 'CHRBTC', 'CHZBTC', 'COTIBTC', 'DEGOBTC', 'DIABTC', 'DOGEBTC', 'DOTBTC', 'DREPBTC', 'DUSKBTC', 'ENJBTC', 'EOSBTC', 'ETHBTC', 'FETBTC', 'FILBTC', 'FTMBTC', 'HBARBTC', 'IOSTBTC', 'JSTBTC', 'KAVABTC', 'KNCBTC', 'LINKBTC', 'LRCBTC', 'LTCBTC', 'LUNABTC', 'MANABTC', 'MATICBTC', 'MDTBTC', 'MITHBTC', 'NEOBTC', 'OGNBTC', 'ONEBTC', 'ONTBTC', 'REEFBTC', 'ROSEBTC', 'RVNBTC', 'SANDBTC', 'SCBTC', 'SOLBTC', 'STMXBTC', 'SUSHIBTC', 'SXPBTC', 'TFUELBTC', 'THETABTC', 'TROYBTC', 'TRXBTC', 'TVKBTC', 'UNIBTC', 'VETBTC', 'WBTCBTC', 'XEMBTC', 'XLMBTC', 'XMRBTC', 'XRPBTC', 'XTZBTC', 'XVGBTC', 'ZILBTC']



def create_and_save_historicals(kline_size: str, start_date: Optional[str] = '1 Feb 2021', end_date: Optional[str] = None, save_compiled: Optional[bool] = False, save_individual: Optional[bool] = False) -> pd.DataFrame:
    """Pools historical information of all coins in BINANCE_SYMBOLS using the Binance API

    Parameters
    ----------
    kline_size : str
        How often the data should be pulled. Options are: 1m, 5m, 1h, 1d
    start_date : str, optional
        The first day of data collection. Format: '%d %b %Y' (defaults is '1 Feb 2021')
    end_date : str, optional
        Last day of data collections. Format: '%d %b %Y' (defaults is today's date)
    save_compiled : bool, optional
        An option to save the full dataframe in a CSV file named 'full_data.csv' (default is False)
    save_individual : bool, optional
        An option to save the individual coins' data in CSV files (default is False)

    Returns
    -------
    pd.DataFrame
        DataFrame with the historical information and statistics of the coins listed in BINANCE_SYMBOLS
    """
    
    df = pd.DataFrame()

    for symbol in BINANCE_SYMBOLS:
        df1 = binance_data_to_df(symbol=symbol, kline_size=frequency, start_date=start_date, end_date=end_date, save=save_individual)
        df1['coin'] = symbol
        df = df.append(df1, True)

    # convert all except the following columns to numeric type
    cols = [i for i in df.columns if i not in ["coin", "close_time"]]

    for col in cols:
        df[col] = pd.to_numeric(df[col])

    df['log_close'] = np.log10(df['close'])
    df['%change'] = (df['close'] - df['open']) / df['open']

    if save_compiled:
        df.to_csv('full_data.csv')

    return df

def binance_data_to_df(symbol: str, kline_size: str, start_date: str, end_date: str, save: bool) -> pd.DataFrame:
    """Helper function for create_and_save_historicals. It queries the Binance API to grab historical information on one coin

    Parameters
    ----------
    symbol : str
        The coin's symbol as listed in Binance
    kline_size : str
        How often the data should be pulled. Options are: '1m', '5m', '1h', '1d'
    start_date : str
        The first day of data collection. Format: '%d %b %Y'
    end_date : str
        Last day of data collections. Format: '%d %b %Y'
    save : bool
        An option to save the coin's data in a CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the historical information of a coin
    """

    filename = '%s-%s-data.csv' % (symbol, kline_size)
    
    # get the start date as oldest_point and newest_point as the last datapoint available
    oldest_point = datetime.strptime(start_date, '%d %b %Y') 

    if end_date is None:
        newest_point = pd.to_datetime(BINANCE_CLIENT.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    else:
        newest_point = datetime.strptime(end_date, '%d %b %Y')

    # calculate number of minutes between start and end point
    delta_min = (newest_point - oldest_point).total_seconds() / 60
    
    #create a bucket for each time segment by dividing total minutes by the corresponding binsize (no. of min in each bucket)
    available_data = math.ceil(delta_min / BINSIZES[kline_size])
    
    if oldest_point == datetime.strptime(start_date, '%d %b %Y'):
        print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: 
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    
    klines = BINANCE_CLIENT.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    
    if save: 
        data.to_csv(os.path.join(os.getcwd(),'data', filename))

    return data


def two_coin_pricing(coin1: str, coin2: str, df: pd.DataFrame) -> Union[pd.Series, pd.Series, pd.Series]:
    """returns the historical log price on two coins and their difference in log prices

    Parameters
    ----------
    coin1 : str
        coin symbol
    coin2 : str
        coin symbol
    df : pd.DataFrame
        DataFrame consisting of historical market pricing
    
    Returns
    -------
    pd.Series, pd.Series, pd.Series
        Three pandas series consisting of log prices of the two coins and the differences in their log prices
    """

    X1 = pd.Series(df[df['coin'] == coin1]['log_close'])
    X1.index = df[df['coin'] == coin1]['close_time']

    X2 = pd.Series(df[df['coin'] == coin2]['log_close'])
    X2.index = df[df['coin'] == coin2]['close_time']
    
    diff = X1.subtract(X2)
    diff.dropna(inplace=True)
    
    return X1, X2, diff


def single_stationarity_test(X: pd.Series, cutoff: Optional[float] = 0.01) -> Union[float, bool]:
    """Tests for time series stationarity using the Augmented Dicky-Fuller Test. Helper function for pair_stationarity_test

    Parameters
    ----------
    X : pd.Series
        time series values of a pandas series (historical coin pricing)
    cutoff : float, optional
        minimum p-value required to be considered statistically significant (default is 0.01)

    Returns
    -------
    float or bool
        returns the p-value if the difference in log pricing is statistically signifant or False if not
    """

    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        return pvalue
    else:
        return False


def pair_stationarity_test(coin1: str, coin2: str, df: pd.DataFrame) -> Union[float, bool]:
    """checks whether a pair of coins is stationary. Helper function for potential pairs

    Parameters
    ----------
    coin1 : str
        coin symbol
    coin2 : str
        coin symbol
    df : pd.DataFrame
        DataFrame consisting of historical market pricing

    Returns
    -------
    float or bool
        returns the p-value if the difference in log pricing is statistically signifant or False if not
    """

    _, _, diff = two_coin_pricing(coin1, coin2, df)
    
    return single_stationarity_test(diff)


def potential_pairs(df: pd.DataFrame, top_n_quartiles: Optional[int] = 2) -> dict:
    """creates a list of potential coins to trade based on stationarity and volume 
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame consisting of historical market pricing
    top_n_quartiles : int, optional
        How many of the top quartiles by volume you want to consider

    Returns
    -------
    list
        list of all potential pairs to trade based
    """

    if top_n_quartiles not in range(1, 5):
        raise IndexError('top_n_quartiles must be between 1 and 4, inclusive')

    volumes = df.groupby(['coin'])['volume'].mean()
    quant_25 = np.quantile(volumes, 0.25)
    quant_50 = np.quantile(volumes, 0.50)
    quant_75 = np.quantile(volumes, 0.75)

    quant_25_index = volumes[volumes <= quant_25].index
    quant_50_index = volumes[volumes <= quant_50][volumes > quant_25].index
    quant_75_index = volumes[volumes <= quant_75][volumes > quant_50].index
    quant_100_index = volumes[volumes > quant_75].index

    index_list = [quant_25_index, quant_50_index, quant_75_index, quant_100_index]
    potential_candidates = {}

    for index in index_list[len(index_list)-top_n_quartiles:]:
        for combo in list(combinations(index, 2)):
            p_value = pair_stationarity_test(combo[0], combo[1], df)
            
            if p_value:
                potential_candidates[combo] = p_value
    
    return potential_candidates


def coin_reversion_volatility(coin1: str, coin2: str, df: pd.DataFrame) -> float:
    """helper function to see how volatile the price differences are - more volatile means more profitable"""
    _, _, diff = two_coin_pricing(coin1, coin2, df)
    
    return diff.std()


def crossing_count(coin1: str, coin2: str, df: pd.DataFrame) -> int:
    """counts the number of times the coins mean revert"""
    
    coin1_price, coin2_price, diff = two_coin_pricing(coin1, coin2, df)
    
    num_crossings = 0
    shifted_diff = list(diff - diff.mean())
    for i in range(1, len(diff)):
        if shifted_diff[i] * shifted_diff[i-1] < 0:
            num_crossings += 1

    return num_crossings


def ranked_crossing(coin_pairs: list, df: pd.DataFrame) -> list:
    """ranks coin pairs from most crossings to fewest in the given dataset"""
    coin_crossing = {}
    
    for coin_pair in coin_pairs:
        coin_crossing[coin_pair] = crossing_count(coin_pair[0], coin_pair[1], df)

    # reverse order so most crossings gets lowest spot in list
    ranked_crossings = sorted(coin_crossing, key=coin_crossing.get, reverse=True)
    
    # assign each pair a number corresponding to its order in the list
    ranked_crossing_dict = dict(zip(ranked_crossings, range(len(ranked_crossings))))

    return ranked_crossing_dict


def ranked_volatility(coin_pairs: list, df: pd.DataFrame) -> list:
    """ranks coin pairs from most volatile to least volatile"""

    crossing_volatility = {}
    
    for coin_pair in coin_pairs:
        crossing_volatility[coin_pair] = coin_reversion_volatility(coin_pair[0], coin_pair[1], df)
    
    # most volatile gets lowest spot
    ranked_volatility = sorted(crossing_volatility, key=crossing_volatility.get, reverse=True)

    # assign each pair a number corresponding to its order in the list
    ranked_volatility_dict = dict(zip(ranked_volatility, range(len(ranked_volatility))))

    return ranked_volatility_dict

def composite_ranking(coin_pairs: list, df: pd.DataFrame) -> list:
    """ranks coin pairs using a combination of volatility and how many times it mean reverts"""
    ranked_crossing_dict = ranked_crossing(coin_pairs, df)
    ranked_volatility_dict = ranked_volatility(coin_pairs, df)

    # sum the two rankings to get a composite score
    full_rank = {}
    for coin_pair in coin_pairs:
        full_rank[coin_pair] = ranked_crossing_dict[coin_pair] + ranked_volatility_dict[coin_pair]

    # sort composite score from smallest to largest - smallest represents the best options to trade
    full_rank_sort = sorted(full_rank, key=full_rank.get, reverse=False)

    return full_rank_sort