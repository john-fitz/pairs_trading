import pandas as pd
import numpy as np
import math
import os
import os.path
import time
from datetime import timedelta, datetime
import pairs_helpers
import pairs_trading
import ast

def relevant_pairs() -> dict:
    with open('data/Anna_coin_pairs.txt') as f:
        lines = f.readlines()

    pairs_dict = {}
    for line in lines:
        key = ""
        vals = []
        
        start_index = 0
        end_index = 0
        for i in range(len(line)):
            if line[i] == '{':
                start_index = i+1
            if line[i] == "}":
                end_index = i
                key = int(line[start_index:end_index])
            if line[i] == "(":
                start_index = i
            if line[i] == ")":
                end_index = i + 1
                vals.append(ast.literal_eval(line[start_index:end_index]))
        pairs_dict[key] = vals

    return pairs_dict

def testing_trading_bot():
    start_time = datetime.now()
    time_holder = datetime.now()
    # 1460 hours is 61 days
    start_period = 1488
    two_months = 1460
    weeks = 0
    day = 0

    full_market_info = pd.read_csv('data/Anna_coins_full_data.csv', index_col=0)

    pairs = relevant_pairs()
    
    times = full_market_info[full_market_info['coin'] == '1INCHBTC']['close_time'].values
    # print(len(times))
    for i in range(start_period, len(times)):
        # to do daily
        if i % 24 == 0:
            # print(f"beginning day {day}")
            day += 1
            print(f"gathering pairs for day {day} of {len(range(start_period, len(times)))//24}")
            
            # information up until the day before to not bias collection of potential pairs
            # previous_info = full_market_info[(full_market_info['close_time'] <= times[i - 24]) & (full_market_info['close_time'] > times[i - two_months])]
            # past info up to current time
            # potential_candidates = list(pairs_helpers.potential_pairs(previous_info, 2).keys())

            # with open('coin_pairs.txt', 'a') as f:
            #     f.write("{" + str(times[i]) + "}" + str(potential_candidates) +"\n")

            potential_candidates = pairs.get(times[i])
            if potential_candidates == None:
                print('not gathering pairs correctly for backtesting')

            # to do weekly
            if i % 168 == 0 and day > 2:
                weeks += 1
                trades = pd.read_csv('testing_trade_log.csv')
                current_profit = sum(trades[(trades['current_position'] == 'closed') & (trades['exit_time'] >= times[i - 168])]['profit'])
                print('week {} profit: {}'.format(weeks, current_profit))
                print(f"week {weeks} time to run: {datetime.now() - time_holder}")
                time_holder = datetime.now()
                
    
        # opening log
        market_info = full_market_info[(full_market_info['close_time'] <= times[i]) & (full_market_info['close_time'] > times[i - two_months])]
        actual_log, fictional_log = pairs_trading.build_trade_log(True)
        potential_trades = pairs_trading.potential_trades_status(potential_candidates, market_info)
        pairs_trading.pseudo_trade(actual_log, fictional_log, potential_trades, market_info, test_mode=True)
    trades = pd.read_csv('testing_trade_log.csv')
    current_profit = sum(trades[trades['current_position'] == 'closed']['profit'])
    print('Total time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
    print('total profit: {}'.format(current_profit))

if __name__ == '__main__':
    testing_trading_bot()