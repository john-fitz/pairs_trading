import pandas as import pd
import numpy as np
from typing import Optional, Union
import pickle

def trade_amount(coin1: str, coin2: str, hedge_ratio: float, log: pd.DataFrame) --> Union[float, float]:
    """ checks positions for coin1 and coin2 to see the dollar amount of each coin that should be traded
    
    Parameters
    ----------
    coin1 : string
        name of coin1 to trade
    coin2 : string
        name of coin2 to trade
    hedge_ratio : float
        ratio of how much to purchase of coin1 to coin2
    log : pd.DataFrame
        DataFrame with logs of past trades

    Returns
    -------
    float, float
        dollar values of coin1 and coin2 to purchase / sell, respectively
    """
    portfolio = portfolio_positions()
    



    update_portfolio(portfolio)
    return (coin1_amt, coin2_amt)



def portfolio_positions() -> dict:
    """returns a dataframe with the portfolio"""
    portfolio = pickle.load( open( "crypto_positions.p", "rb" ) )
    
    return portfolio


def update_portfolio(portfolio: dict) -> None:
    """takes the portfolio as a dictionary and saves it as a pickle file"""

    pickle.dump( portfolio, open( "crypto_positions.p", "wb" ) )