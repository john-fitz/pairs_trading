import pandas as pd
import numpy as np
from typing import Optional, Union
import pickle

DEFAULT_PURCHASE_AMT = 100

def trade_amount(coin1: str, coin2: str, hedge_ratio: float, long1: bool, fictional: bool) -> Union[float, float]:
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
    portfolio = portfolio_positions(fictional=fictional)
    
    remaining_balance = portfolio['balance']
    max_budget = 0.90 * remaining_balance # trying not to exhaust all money due to slippage
    coin1_coin_amt, coin1_dollar_amt = portfolio[coin1]
    coin2_coin_amt, coin2_dollar_amt = portfolio[coin2]

    # if we are going long on coin1, the limiting factor is how much of coin2 we have left to sell and cash reserves
    if long1:
        # limiting max trade to 10% of remaining balance
        coin2_dollar_amt *= 0.10
        # just for wiggle room, we leave 10% of balance in reserves
        if coin2_dollar_amt * hedge_ratio >= max_budget:
            min_amt = max_budget
        else:
            min_amt = coin2_dollar_amt
    else:
        # limiting max trade to 10% of remaining balance
        coin1_dollar_amt *= 0.10
        # just for wiggle room, we leave 10% of balance in reserves
        if coin1_dollar_amt >= max_budget:
            min_amt = max_budget
        else:
            min_amt = coin1_dollar_amt
    if coin1_dollar_amt ==0 or coin2_dollar_amt == 0:
        return (0, 0)
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


def save_portfolio(fictional: bool, portfolio: dict) -> None:
    """takes the portfolio as a dictionary and saves it as a pickle file"""
    file_name = portfolio_name(fictional=fictional)
    pickle.dump( portfolio, open( file_name, "wb" ) )


def portfolio_name(fictional: bool) -> str:
    """returns the proper name (as string) of the portfolio"""
    return "fictional_crypto_positions.p" if fictional else "actual_crypto_positions.p"


def update_portfolio_positions(market_info: pd.DataFrame,) -> None:
    """Goes through the portfolio and updates values based on coin prices"""
    actual_portfolio = portfolio_positions(fictional=False)
    fictional_portfolio = portfolio_positions(fictional=True)

    fictional = False
    for portfolio in [actual_portfolio, fictional_portfolio]:
        for coin in portfolio.keys():
            if coin != 'balance':
                # try:
                coin_price = market_info[market_info['coin'] == coin]['close'].iloc[-1]
                coin_amt = portfolio[coin][0]
                portfolio[coin] = (coin_amt, coin_amt*coin_price)
                # except:
                #     print("can't find information for " + str(coin))
        save_portfolio(fictional=fictional, portfolio=portfolio)
        fictional = True

    return None

def purchase_initial_position(coin_pairs: str, market_info: pd.DataFrame) -> None:
    """loops through coin pairs and if there are any coins not in the portfolio, it add them"""
    actual_portfolio = portfolio_positions(fictional=False)
    fictional_portfolio = portfolio_positions(fictional=True)

    coin_list = []
    for coin_pair in coin_pairs:
        coin1 = coin_pair[0]
        coin2 = coin_pair[1]
        if coin1 not in coin_list:
            coin_list.append(coin1)
        if coin2 not in coin_list:
            coin_list.append(coin2)

    fictional = False
    for portfolio in [actual_portfolio, fictional_portfolio]:
        for coin in coin_list:
            if coin not in portfolio.keys() and portfolio['balance'] > 0:
                print(f"adding ${DEFAULT_PURCHASE_AMT} of {coin} to {'fictional' if fictional else 'actual'} portfolio")
                # print(len(market_info[market_info['coin'] == coin]['close']))
                coin_price = market_info[market_info['coin'] == coin]['close'].iloc[-1]
                portfolio[coin] = (DEFAULT_PURCHASE_AMT / coin_price, coin_price)
                portfolio['balance'] -= DEFAULT_PURCHASE_AMT
        save_portfolio(fictional=fictional, portfolio=portfolio)
        fictional=True

def portfolio_value(fictional: Optional[bool]=False) -> float:
    portfolio = portfolio_positions(fictional=fictional)
    portfolio_value = portfolio['balance']
    for coin in portfolio.keys():
        if coin != 'balance':
            portfolio_value += portfolio[coin][1]
    
    return portfolio_value