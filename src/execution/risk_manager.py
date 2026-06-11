import pandas as pd
import numpy as np

def calculate_dynamic_risk(current_price, atr_value, direction, risk_reward_ratio=2.0):
    """
    Calculates dynamic Stop Loss and Take Profit based on ATR and direction.

    Args:
        current_price (float): The current entry price.
        atr_value (float): The current ATR value.
        direction (int): 1 for Buy/Up, 0 for Sell/Down.
        risk_reward_ratio (float): The target reward ratio against 1x ATR risk.

    Returns:
        tuple: (stop_loss_price, take_profit_price)
    """
    # Base risk is 1x ATR
    risk_amount = atr_value

    # Target reward is risk_reward_ratio x ATR (default is 2.0 -> 2x ATR)
    reward_amount = atr_value * risk_reward_ratio

    if direction == 1:
        # BUY trade
        sl_price = current_price - risk_amount
        tp_price = current_price + reward_amount
    else:
        # SELL trade
        sl_price = current_price + risk_amount
        tp_price = current_price - reward_amount

    return sl_price, tp_price

def calculate_position_size(account_balance, risk_percent, entry_price, sl_price, contract_size=100000):
    """
    Calculates the position size (in lots) based on account risk.
    """
    risk_amount_currency = account_balance * (risk_percent / 100.0)

    # Difference in price points
    price_risk = abs(entry_price - sl_price)

    if price_risk == 0:
        return 0.01 # Minimum lot size

    # Standard formula for lot size in Forex: Risk / (SL_Pips * Pip_Value)
    # Simplified generic calculation:
    volume = risk_amount_currency / (price_risk * contract_size)

    # Round to nearest 0.01 lots
    volume = max(0.01, round(volume, 2))
    return volume
