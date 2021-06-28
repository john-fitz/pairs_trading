"""Contains functions to visualize the outputs of the pairs trading algorithm and its status

This file can be imported as a module and contains the following functions:

    * plot_profit - plots the returns in dollar amounts
    * plot_coin_crossings - plots the difference in log price between the two coins and adds markers for when they were bought into/sold
    * ms_to_dates - takes a list of times in ms and converts to a list of dates
    * date_to_ms - takes a list of dates and converts to a list of times in ms
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pairs_helpers
from typing import Optional

def plot_profit(current_log: pd.DataFrame) -> None:    
    """Takes the trade log and plots the returns in dollar amounts"""
    profit = []
    times = []
    sum_profit = 0

    for _, row in current_log.iterrows():
        sum_profit += row['profit']
        if row['exit_time'] not in times:
            profit.append(sum_profit)
            times.append(row['exit_time'])
    
    dates = []
    for time in times:
        s = time/1000
        try:
            converted = datetime.fromtimestamp(s).strftime('%Y-%m-%d')
        except:
            pass
        dates.append(converted)
        
    plt.figure(figsize=(15,10))
    plt.plot(dates, profit)
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.xticks(dates[::40], rotation=40)
    
    plt.show();

def plot_coin_crossings(coin1: str, coin2: str, df: pd.DataFrame, log: pd.DataFrame, start_time: Optional[float]=None, end_time: Optional[float]=None) -> None:
    """ plots the difference in log price between the two coins and adds markers for when they were bought into/sold
    
    start_time : float, optional 
        start of the range of values to plot in ms (default is None)
    end_time : float, optional
        end of the range of values to plot in ms (default is None)
    """

    if start_time:
        df = df[(df['close_time'] > start_time) & (df['close_time'] <= end_time)]
    _, _, diff = pairs_helpers.two_coin_pricing(coin1, coin2, df)
    coin_logs = log[(log['coin1'] == coin1) & (log['coin2'] == coin2)]
    buy_times = coin_logs['entry_time']
    sell_times = coin_logs['exit_time']
    diff_buy = diff[diff.index.isin(buy_times)]
    diff_sell = diff[diff.index.isin(sell_times)]
    diff_std = diff.rolling(30).std()
    diff_mean = coin_logs['exit_mean']
    diff_mean.index = coin_logs['entry_time']
    s = np.empty(len(diff))
    s[:] = 0
    s = pd.Series(s)
    s.index = diff.index
    diff_mean = s + diff_mean
    diff_mean = diff_mean.fillna(method='ffill')
    diff_plus = diff_mean + diff_std
    diff_minus = diff_mean - diff_std
    plt.figure(figsize=(15,10))
    plt.plot(ms_to_dates(diff.index), diff)
    plt.plot(ms_to_dates(diff_buy.index), diff_buy, '^', markersize=6, color='g')
    plt.plot(ms_to_dates(diff_sell.index), diff_sell, 'v', markersize=6, color='r')
    plt.plot(ms_to_dates(diff_mean.index), diff_mean, 'k-')
    plt.plot(ms_to_dates(diff_plus.index), diff_plus, 'r--')
    plt.plot(ms_to_dates(diff_minus.index), diff_minus, 'b--')
    plt.xticks(ms_to_dates(diff.index)[::200], rotation=40)

    plt.show();

def ms_to_dates(times: list) -> list:
    """takes a list of times in ms and converts to a list of dates"""
    dates = []
    for time in times:
        s = time / 1000
        try:
            converted = datetime.fromtimestamp(s).strftime('%Y-%m-%d')
        except:
            pass
        dates.append(converted)

    return dates

def date_to_ms(date: str) -> list:
    """takes a list of dates and converts to a list of times in ms"""
    dt_obj = datetime.strptime(date, '%Y-%m-%d')
    millisec = dt_obj.timestamp() * 1000

    return millisec