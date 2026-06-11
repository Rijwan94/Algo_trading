try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from src.data.market_data_mt5 import init_mt5

def place_order(symbol, direction, volume, sl_price, tp_price, magic_number=123456):
    """
    Executes a trade on MT5 with the specified parameters.

    Args:
        symbol (str): The symbol to trade (e.g., "EURUSD").
        direction (int): 1 for Buy, 0 for Sell.
        volume (float): Position size in lots.
        sl_price (float): Stop loss price.
        tp_price (float): Take profit price.
        magic_number (int): Unique identifier for the bot's trades.
    """
    if not MT5_AVAILABLE:
        print(f"SIMULATION: Placing order - Symbol: {symbol}, Dir: {'BUY' if direction==1 else 'SELL'}, Vol: {volume}, SL: {sl_price}, TP: {tp_price}")
        return True

    if not init_mt5():
        return False

    # Get symbol properties
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{symbol} not found.")
        mt5.shutdown()
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed.")
            mt5.shutdown()
            return False

    # Determine order type and price
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid

    # Prepare trade request dict
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20, # Max slippage
        "magic": magic_number,
        "comment": "ML Bot Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send trading request
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}")
        print(result)
        mt5.shutdown()
        return False

    print(f"Order successfully placed. Ticket: {result.order}")
    mt5.shutdown()
    return True

def close_all_positions(magic_number=123456):
    """
    Closes all active positions opened by this bot.
    """
    if not MT5_AVAILABLE:
        print("SIMULATION: Closing all positions.")
        return True

    if not init_mt5():
        return False

    positions = mt5.positions_get(magic=magic_number)
    if positions is None or len(positions) == 0:
        mt5.shutdown()
        return True

    for pos in positions:
        symbol = pos.symbol
        volume = pos.volume
        pos_type = pos.type
        ticket = pos.ticket

        if pos_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": magic_number,
            "comment": "ML Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        mt5.order_send(request)

    mt5.shutdown()
    return True
