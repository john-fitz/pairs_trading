import pandas as import pd
import numpy as np
from typing import Optional, Union
import pickle

def trade_amount(coin1: str, coin2: str, hedge_ratio: float, long1: bool) --> Union[float, float]:
    """ checks positions for coin1 and coin2 to see the dollar amount of each coin that should be traded

    Assumes that the limiting factor is the one that we have to sell and can buy up to the remaining balance on our account.
    If there is no money to trade or no ability left to sell a coin, the function returns (0,0)
    
    Parameters
    ----------
    coin1 : string
        name of coin1 to trade
    coin2 : string
        name of coin2 to trade
    hedge_ratio : float
        ratio of how much to purchase of coin1 to coin2
    short1 : bool
        flag if we are going long on coin1

    Returns
    -------
    float, float
        dollar values of coin1 and coin2 to purchase / sell, respectively
    """
    portfolio = portfolio_positions()
    
    remaining_balance = portfolio['remaining_balance']
    max_budget = 0.90 * remaining_balance # trying not to exhaust all money due to slippage
    coin1_coin_amt, coin1_dollar_amt = portfolio[coin1]
    coin2_coin_amt, coin2_dollar_amt = portfolio[coin2]

    # if we are going long on coin1, the limiting factor is how much of coin2 we have left to sell and cash reserves
    if long1:
        # limiting max trade to 10% of remaining balance
        coin2_dollar_amt *= 0.10
        # just for wiggle room, we leave 10% of balance in reserves
        if coin2_dollar_amt * hedge_ratio >= max_budget
            min_amt = max_budget
        else:
            min_amt = coin2_dollar_amt
    else:
        # limiting max trade to 10% of remaining balance
        coin1_dollar_amt *= 0.10
        # just for wiggle room, we leave 10% of balance in reserves
        if coin1_dollar_amt >= max_budget
            min_amt = max_budget
        else:
            min_amt = coin1_dollar_amt

    coin1_amt = (min_amt / coin1_dollar_amt) * coin1_coin_amt
    coin2_amt = (min_amt / coin2_dollar_amt) * coin1_coin_amt * hedge_ratio

    return (coin1_amt, coin2_amt)



def portfolio_positions(fictional: bool) -> dict:
    """returns a dataframe with the portfolio"""
    # dictionary is of the form {coin_name: (number_of_coins_owned, dollar_value_of_position)} with the 
    # first entry being {remaining_balance: float}
    file_name = portfolio_name(fictional=fictional)
    portfolio = pickle.load( open( file_name, "rb" ) )
    
    return portfolio


def update_portfolio(fictional: bool, portfolio: dict) -> None:
    """takes the portfolio as a dictionary and saves it as a pickle file"""
    file_name = portfolio_name(fictional=fictional)
    pickle.dump( portfolio, open( file_name, "wb" ) )


def portfolio_name(ficitonal: bool) -> str:
    """returns the proper name (as string) of the portfolio"""
    return "fictional_crypto_positions.p" if fictional else "actual_crypto_positions.p"
