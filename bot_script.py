import pandas as pd
import numpy as np
import math
import os
import os.path
import time
from datetime import timedelta, datetime
import pairs_helpers
import pairs_trading


def testing_trading_bot():
    start_time = datetime.now()
    # 1460 hours is 61 days
    start_period = 1488
    two_months = 1460
    weeks = 0
    day = 0
    pairs = pd.DataFrame()

    full_market_info = pd.read_csv('data/full_data.csv', index_col=0)
    if len(full_market_info) == 0:
        print('not reading in csv correctly')
        return None
    
    times = full_market_info[full_market_info['coin'] == 'SANDBTC']['close_time'].values
    for i in range(start_period, len(times)):
        # to do daily
        if i % 24 == 0:
            print(f"beginning day {day}")
            day += 1
            # print(f"beginning week {weeks}")
            # weeks += 1

            # information up until the day before to not bias collection of potential pairs
            previous_info = full_market_info[(full_market_info['close_time'] <= times[i - 24]) & (full_market_info['close_time'] > times[i - two_months])]
            
            # past info up to current time
            market_info = full_market_info[full_market_info['close_time'] > times[i - two_months]]
            potential_candidates = list(pairs_helpers.potential_pairs(previous_info, 2).keys())
            last_time = previous_info['close_time'].iloc[-1]
            with open("pairs.txt", "a") as myfile:
                myfile.write("{" + str(last_time) + "}: " + str(potential_candidates) + "\n")
            

            # to do weekly
            if i % 168 == 0 and day > 1:
                weeks += 1
                trades = pd.read_csv('testing_trade_log.csv')
                current_profit = sum(trades[(trades['current_position'] == 'closed') & (trades['exit_time'] >= times[i - 168])]['profit'])
                print('week {} profit: {}'.format(weeks, current_profit))
    
        #opening log
        actual_log, fictional_log = pairs_trading.build_trade_log(True)
        potential_trades = pairs_trading.potential_trades_status(potential_candidates, market_info)
        pairs_trading.pseudo_trade(actual_log, fictional_log, potential_trades, market_info, test_mode=True)
    
    trades = pd.read_csv('testing_trade_log.csv')
    current_profit = sum(trades[trades['current_position'] == 'closed']['profit'])
    pairs.to_csv('pairs_over_time.csv', index=False)
    print('Total time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
    print('total profit: {}'.format(current_profit))

if __name__ == '__main__':
    testing_trading_bot()