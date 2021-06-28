"""Trading logic for pairs trading algorithm

Many of the functions are helpers for the pseudo_trade function

This file can be imported as a module and contains the following functions:

    * potential_trades_status - df of potential coins and determines whether they should be traded
    * pseudo_trade - implements logic of buying/selling and executes it by updating the log
    * build_trade_log - returns the trade log if a CSV is found, otherwise it creates a new DataFrame
"""


import pandas as pd
import numpy as np
from datetime import datetime
import pairs_helpers
import time
import visualizations
from typing import Optional, Union


PORTFOLIO_VALUE = 10000
TRADE_AMT_DEFAULT = PORTFOLIO_VALUE * 0.05
MAX_LOSS = -1 * (PORTFOLIO_VALUE * 0.01)

def potential_trades_status(coin_pairs: list, df: pd.DataFrame) -> pd.DataFrame:
    """Goes through the list of potential coins and determines whether they should be traded

    Parameters
    ----------
    coin_pairs : list
        list of potential pairs of coins to trade
    df : pd.DataFrame
        DataFrame consisting of historical market pricing

    Returns
    -------
    pd.DataFrame
        collection of information on the coins listed including whether they meet the threshold to open a trading position
    """

    potential_trades_columns = ['coin1', 'coin2', 'coin1_price', 'coin2_price', 'mean', 'stdev', 'coin1_long', 'coin2_long', 'current_condition', 'hedge_ratio']
    potential_trades = pd.DataFrame(columns = potential_trades_columns)

    for coin_pair in coin_pairs:
        row_vals = {}
        row_vals['coin1'] = coin_pair[0]
        row_vals['coin2'] = coin_pair[1]

        coin1_pricing, coin2_pricing, diff = pairs_helpers.two_coin_pricing(coin_pair[0], coin_pair[1], df)
        current_diff = diff.values[-1]
        row_vals['coin1_price'] = np.exp(coin1_pricing.values[-1])
        row_vals['coin2_price'] = np.exp(coin2_pricing.values[-1])

        mean = np.mean(diff)
        stdev = np.std(diff, ddof=1)
        row_vals['mean'] = mean
        row_vals['stdev'] = stdev

        # if the difference is greater than 1 std but less than 2, we should trade on it
        row_vals['coin1_long'] = current_diff >= (mean + stdev) and current_diff < 2*(mean + stdev) 
        row_vals['coin2_long'] = current_diff <= (mean - stdev) and current_diff > 2*(mean - stdev)
        row_vals['current_condition'] = 'above' if current_diff >= mean else 'below'
        
        #TODO - CALCULATE HEDGE RATIOS
        row_vals['hedge_ratio'] = 1

        potential_trades = potential_trades.append(row_vals, ignore_index=True)

    return potential_trades

def update_open_positions(log: pd.DataFrame, potential_buys: pd.DataFrame, full_market_info: pd.DataFrame, fictional: bool, test_mode: Optional[bool]=False, test_time: Optional[float]=None) -> pd.DataFrame:
    """Grabs the open positions from the log, updates them and adds logic whether they should be sold

    Parameters
    ----------
    log : pd.DataFrame
        trading log
    potential_buys : pd.DataFrame
        information on potential trades to open - output from potential_trades_status function
    full_market_info : pd.DataFrame
        DataFrame consisting of historical market pricing
    fictional : bool
        whether teh log provided is fictional or actual
    test_mode : bool, optional
        flag for whether this is being run in production or for testing (default option is False)
    test_time:
        time needed to know how long the position as been open. Only needed if test_mode is true (default option is None)

    Returns
    -------
    pd.DataFrame
        updated information on prices, profit, whether they should sell, etc for open positions
    """

    open_positions = log[log['current_position'] == 'open']
    
    for index, row in open_positions.iterrows():
        row_info = row.to_dict()
        
        # difference between open and now in ms so divide by number of ms in a day to get # of days
        if test_mode:
            now = test_time
        else:
            now = round(time.time() * 1000)
        
        open_days = (now - row_info['entry_time']) // 86400000
        open_positions.loc[index, 'open_days'] = open_days
        coin1 = row_info['coin1']
        coin2 = row_info['coin2']
        
        coin1_pricing, coin2_pricing, diff = pairs_helpers.two_coin_pricing(coin1, coin2, full_market_info)
        current_diff = diff.values[-1]
        coin1_price = np.exp(coin1_pricing.values[-1])
        coin2_price = np.exp(coin2_pricing.values[-1])
        open_positions.loc[index, 'coin1_price'] = coin1_price
        open_positions.loc[index, 'coin2_price'] = coin2_price
        
        short1_hedge = -row_info['hedge_ratio'] if row_info['coin1_long'] == False else 1
        short2_hedge = -row_info['hedge_ratio'] if row_info['coin2_long'] == False else 1

        profit1 = (row_info['coin1_entry_price'] - coin1_price) * row_info['coin1_amt'] * short1_hedge
        profit2 = (row_info['coin2_entry_price'] - coin2_price) * row_info['coin2_amt'] * short2_hedge
        current_profit = profit1 + profit2
        open_positions.loc[index, 'profit'] = current_profit
        
        # if not cointegratred anymore, we shouldn't hold onto it
        relevant_positions1 = potential_buys[((potential_buys['coin1'] == coin1) & (potential_buys['coin2'] == coin2))]
        relevant_positions2 = potential_buys[((potential_buys['coin1'] == coin2) & (potential_buys['coin2'] == coin1))]

        not_cointegrated = len(relevant_positions1) + len(relevant_positions2) == 0 
        
        # need to identify positive if crosses back over threshold (entry_condition)
        condition = 'above' if current_diff >= row_info['exit_mean'] else 'below'
        
        if open_days >= 10 or current_profit < MAX_LOSS or row_info['entry_condition'] != condition: #or not_cointegrated:
            open_positions.loc[index, 'suggested_move'] = 'sell'

            if open_days >= 10:
                open_positions.loc[index, 'sell_reason'] = 'exceeds hold period'
            elif current_profit < MAX_LOSS:
                open_positions.loc[index, 'sell_reason'] = 'stop loss'

            elif row_info['entry_condition'] != condition:
                open_positions.loc[index, 'sell_reason'] = 'mean reverted'
            # elif not_cointegrated:
            #     open_positions.loc[index, 'sell_reason'] = 'no longer cointegrated'

    open_positions.dropna(subset = ["coin1"], inplace=True)    
    open_positions.to_csv(log_name(fictional=fictional, test_mode=test_mode, open_position=True), index=False)
    return open_positions


def update_log(log: pd.DataFrame, open_positions: pd.DataFrame, fictional: bool, test_mode: bool) -> None:
    """Method to update the trade log for all open positions - inlcuding whether they were sold

    In testing, this step is equivalent to executing the trade.

    Parameters
    ---------
    log : pd.DataFrame
        trade log
    open_positions : pd.DataFrame
        information on all open positions
    fictional : boolean
        if the log is the fictional log
    test_mode : bool, optional
        flag for testing so it knows which log to save to

    Returns
    -------
    None
        updates the log, saves it, but doesn't return anything of value
    """

    for index, row in open_positions.iterrows():
        row_info = row.to_dict()
        coin1 = open_positions.loc[index, 'coin1']
        coin2 = open_positions.loc[index, 'coin2']

        log_row = log[(log['coin1'] == coin1) & (log['coin2'] == coin2) & (log['current_position'] == 'open')]
        if len(log_row) == 0:
            log = log.append(row_info, ignore_index=True)
        else:
            row_index = log[(log['coin1'] == coin1) & (log['coin2'] == coin2) & (log['current_position'] == 'open')].index.values.astype(int)[0]
            log = log.drop(index=row_index)
            log = log.append(row_info, ignore_index=True)

        
        log.dropna(subset = ["coin1"], inplace=True)

    log.to_csv(log_name(fictional=fictional, test_mode=test_mode, open_position=False), index=False)
    # print(f"updated {log_name(fictional=fictional, test_mode=test_mode)}")
    return None

def pseudo_trade(actual_log: pd.DataFrame, fictional_log: pd.DataFrame, potential_trades: pd.DataFrame, full_market_info: pd.DataFrame, trade_amt: Optional[float]=TRADE_AMT_DEFAULT, test_mode: Optional[bool]=False) -> None:
    """Implements logic of buying/selling and executes it by updating the log
    
    Parameters
    ----------
    actual_log : pd.DataFrame
        actual trade log of executed trades
    fictional_log : pd.DataFrame
        fictional trade log that does not stop
    potential_trades : pd.DataFrame
        information on all potential pairs of coins that we could trade on
    full_market_info : pd.DataFrame
        DataFrame consisting of historical market pricing
    test_mode : bool, optional
        flag for whether this is being run in production or for testing (default option is False)
    trade_amt : float, optional
        how much money to place on each trade in the long direction (default is $50)
    test_mode : bool, optional
        flag for whether we are in testing vs live

    Returns
    -------
    None
        updates the logs but doesn't return anything of value
    """

    if not test_mode:
        actual_log, fictional_log = build_trade_log()
        potential_trades = potential_trades_status(potential_candidates, market_info)
        full_market_info = pairs_helpers.create_and_save_historicals(start_date='1 Jun 2020', save_compiled=True, save_individual=False)
        test_time = None
    else:
        test_time = full_market_info['close_time'].iloc[-1]
    
    potential_buys = potential_trades[(potential_trades['coin1_long'] == True) | (potential_trades['coin2_long'] == True)]
    
    actual_open_positions = update_open_positions(actual_log, potential_trades, full_market_info, fictional=False, test_mode=test_mode, test_time=test_time)
    fictional_open_positions = update_open_positions(fictional_log, potential_trades, full_market_info, fictional=True, test_mode=test_mode, test_time=test_time)
    # print('inside trading function')

    # identify buys
    halt_actual = halt_actual_trading(fictional_log)
    if halt_actual:
        print('halting opening positions due to poor performance')
    for index, row in potential_buys.iterrows():
        row_info = row.to_dict()
        
        # if we should buy it, then check and execute trade
        if row_info['coin1_long'] or row_info['coin2_long']:
            # identify positions with those coins. Checks both entries in case the coins are flipped            
            relevant_actual_positions = actual_open_positions[((actual_open_positions['coin1'] == row_info['coin1']) & 
                                              (actual_open_positions['coin2'] == row_info['coin2'])) | 
                                              ((actual_open_positions['coin1'] == row_info['coin2']) & 
                                              (actual_open_positions['coin1'] == row_info['coin1']))]
            
            relevant_fictional_positions = fictional_open_positions[((fictional_open_positions['coin1'] == row_info['coin1']) & 
                                              (fictional_open_positions['coin2'] == row_info['coin2'])) | 
                                              ((fictional_open_positions['coin1'] == row_info['coin2']) & 
                                              (fictional_open_positions['coin1'] == row_info['coin1']))]

            # if we haven't traded those coins in the past or all past positions are closed
            if len(relevant_fictional_positions) == 0 or len(relevant_actual_positions) == 0:
                trade = {}
                trade['coin1'] = row_info['coin1']
                trade['coin2'] = row_info['coin2']
                trade['entry_condition'] = row_info['current_condition'] 
                trade['exit_mean'] = row_info['mean']
                
                # redo later when figure out hedge amounts
                coin1_price = row_info['coin1_price']
                coin2_price = row_info['coin2_price']
                coin1_amt = trade_amt / coin1_price
                coin2_amt = trade_amt / coin2_price
                
                trade['coin1_amt'] = coin1_amt * (row_info['coin1_long'] + (1-row_info['coin1_long']) / row_info['hedge_ratio'])
                trade['coin2_amt'] = coin2_amt * (row_info['coin2_long'] + (1-row_info['coin2_long']) * row_info['hedge_ratio'])
                trade['coin1_long'] = row_info['coin1_long']
                trade['coin2_long'] = row_info['coin2_long']
                trade['coin1_entry_price'] = coin1_price
                trade['coin2_entry_price'] = coin2_price
                trade['coin1_exit_price'] = None
                trade['coin2_exit_price'] = None
                trade['entry_time'] = full_market_info[full_market_info['coin'] == row_info['coin1']]['close_time'].iloc[-1]
                trade['open_day_count'] = 0
                trade['exit_time'] = None
                trade['current_position'] = 'open'
                trade['profit'] = 0
                trade['suggested_move'] = 'hold'
                trade['hedge_ratio'] = row_info['hedge_ratio']
                trade['sell_reason'] = None

                print('opening position for {} and {}'.format(trade['coin1'], trade['coin2']))
                # if row_info['coin1_long']:
                #     print('going long {} units of {} at ${} and short {} units of {} at ${}'.format(trade['coin1_amt'], trade['coin1'], coin1_price, trade['coin2_amt'], trade['coin2'], coin2_price))
                # else:
                #     print('going long {} units of {} at ${} and short {} units of {} at ${}'.format(trade['coin2_amt'], trade['coin2'], coin2_price, trade['coin1_amt'], trade['coin1'], coin1_price))
               
                if not halt_actual and len(relevant_actual_positions) == 0:
                    actual_log = actual_log.append(trade, ignore_index=True)
                
                if len(relevant_fictional_positions) == 0:
                    fictional_log = fictional_log.append(trade, ignore_index=True)
        
    # sell
    for open_positions in [actual_open_positions, fictional_open_positions]:
        for index, row in open_positions.iterrows():
            row_info = row.to_dict()
            if row_info['suggested_move'] == 'sell':
                print('closing position for  {} and {}'.format(row_info['coin1'], row_info['coin2']))
                # if row_info['coin1_long']:
                #     print('closing long positions of {} units of {} at ${} and short position of {} units of {} at ${}'.format(row_info['coin1_amt'], row_info['coin1'], row_info['coin1_exit_price'], row_info['coin2_amt'], row_info['coin2'], row_info['coin2_exit_price']))
                # else:
                #     print('closing long positions of {} units of {} at ${} and short position of {} units of {} at ${}'.format(row_info['coin2_amt'], row_info['coin2'], row_info['coin2_exit_price'], row_info['coin1_amt'], row_info['coin1'], row_info['coin1_exit_price']))

                open_positions.loc[index, 'coin2_exit_price'] = row_info['coin2_price']
                open_positions.loc[index, 'coin1_exit_price'] = row_info['coin1_price']
                open_positions.loc[index, 'exit_time'] = test_time if test_mode else round(time.time() * 1000)
                open_positions.loc[index, 'current_position'] = 'closed'
                                                                
    update_log(log=actual_log, open_positions=actual_open_positions, fictional=False, test_mode=test_mode)
    update_log(log=fictional_log, open_positions=fictional_open_positions, fictional=True, test_mode=test_mode)

    return None
    


def halt_actual_trading(fictional_log: pd.DataFrame) -> bool:
    """reads through the fictional trade log and halts if we are in a period of continuous losses under ideal scenarios"""
    week_ago = time.time() * 1000 - 604800000
    last_week_trades = fictional_log[fictional_log['entry_time'] >= week_ago]
    
    if len(last_week_trades) == 0:
        return False

    bad_trades = 0
    num_trades = 0
    for _, row in last_week_trades.iterrows():
        num_trades += 1
        entry_amt = row['coin1_entry_price'] * row['coin1_amt'] + row['coin2_entry_price'] * row['coin2_amt'] 
        loss_pct = row['profit'] / entry_amt
        
        if loss_pct >= 0.5:
            bad_trades += 1
    
    pct_bad = bad_trades / num_trades
    return True if pct_bad >= 0.5 else False


def build_trade_log(test_mode: Optional[bool]=False) -> Union[pd.DataFrame, pd.DataFrame]:
    """returns the trade log if a CSV is found, otherwise it creates a new DataFrame to return"""
    try:
        actual_log = pd.read_csv(log_name(fictional=False, test_mode=test_mode, open_position=False))
        fictional_log = pd.read_csv(log_name(fictional=True, test_mode=test_mode, open_position=False))
    
    except:
        trade_log_columns = ['coin1', 'coin2', 'entry_condition', 'exit_mean', 'coin1_amt', 'coin2_amt', 'coin1_long', 'coin2_long',
                    'coin1_entry_price', 'coin2_entry_price', 'coin1_exit_price', 'coin2_exit_price', 'entry_time', 'hedge_ratio',
                    'open_day_count', 'exit_time', 'current_position', 'suggested_move', 'profit', 'sell_reason']

        actual_log = pd.DataFrame(columns=trade_log_columns)
        fictional_log = pd.DataFrame(columns=trade_log_columns)
    
    return actual_log, fictional_log

def log_name(fictional: bool, test_mode: bool, open_position: bool) -> str:
    """determines the correct csv title for the log"""
    log_title = ""
    if fictional:
        log_title += 'fictional_'
    if test_mode:
        log_title += 'testing_'
    if open_position:
        log_title += 'open_positions.csv'
    else:
        log_title += 'trade_log.csv'
    return log_title
