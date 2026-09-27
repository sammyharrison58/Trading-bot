import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
import csv

# ====================== CONFIG ======================
LOGIN = int(os.getenv("XM_LOGIN", "168710028"))
PASSWORD = os.getenv("XM_PASSWORD", "Harrison11654$")
SERVER = os.getenv("XM_SERVER", "XMGlobalKE-MT5 2")
MT5_PATH = os.getenv("MT5_PATH", "")

SYMBOLS = ["EURUSD", "GBPUSD", "GOLD", "US30", "US500"]  # Add stock CFDs carefully

TIMEFRAME       = mt5.TIMEFRAME_H1
FAST_EMA        = 9
SLOW_EMA        = 21
TREND_EMA       = 50
TREND_TIMEFRAME = mt5.TIMEFRAME_H4

RISK_PERCENT            = 0.6          # Risk per trade (% of equity)
ATR_PERIOD              = 14
ATR_SL_MULTIPLIER       = 1.8
ATR_TP_MULTIPLIER       = 3.2
TRAIL_START_ATR         = 1.2          # Start trailing after 1.2 × ATR profit
TRAIL_STEP_ATR          = 0.8          # Trail by 0.8 × ATR
BREAKEVEN_TRIGGER_R     = 1.0          # Move to BE after 1R

MAX_OPEN_TRADES         = 4
MAX_DAILY_LOSS_PERCENT  = 2.5
MAGIC                   = 20250926
CHECK_SECONDS           = 30

# Telegram (optional - leave empty to disable)
TELEGRAM_TOKEN = ""                    # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = ""                  # e.g. "987654321"

# Trading hours (server time). None = almost 24/5
TRADING_HOURS = (6, 21)
# ====================================================

# Logging setup
log_path = Path("xm_advanced_bot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger()

# Trade log CSV
TRADE_LOG = Path("trades_log.csv")
if not TRADE_LOG.exists():
    with open(TRADE_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "symbol", "type", "volume", "price", "sl", "tp", "ticket", "comment"])

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")

def connect() -> bool:
    init_attempts = []

    if LOGIN and PASSWORD and SERVER:
        init_attempts.append({
            "login": LOGIN,
            "password": PASSWORD,
            "server": SERVER,
        })
    if MT5_PATH:
        last = init_attempts[-1] if init_attempts else {}
        last = dict(last)
        last["path"] = MT5_PATH
        init_attempts = [last] if init_attempts else [{"path": MT5_PATH}]

    # If MetaTrader terminal is already logged in and open, initialize() without
    # explicit credentials can reuse the active terminal session.
    init_attempts.append({})

    for init_args in init_attempts:
        try:
            mt5.shutdown()
        except Exception:
            pass

        if not mt5.initialize(**init_args):
            error = mt5.last_error()
            logger.warning("Init attempt failed: %s | args=%s", error, init_args)
            continue

        acc = mt5.account_info()
        if acc is not None:
            logger.info(f"Connected | Login: {acc.login} | Balance: {acc.balance:.2f} | Equity: {acc.equity:.2f} {acc.currency}")
            send_telegram(f"🤖 Bot started\nBalance: {acc.balance:.2f} {acc.currency}")
            return True

        logger.warning("Initialization succeeded but account_info() is None. The terminal is not logged in or no account is active.")
        try:
            terminal = mt5.terminal_info()
            logger.warning("Terminal info: %s", terminal)
        except Exception:
            pass

    logger.error("Init failed: %s | login=%s | server=%s | mt5_path=%s", mt5.last_error(), LOGIN, SERVER, MT5_PATH or "auto-detect")
    return False

def get_rates(symbol, timeframe, bars=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = np.abs(df["high"] - df["close"].shift())
    lc = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def get_signal(symbol):
    df = get_rates(symbol, TIMEFRAME, 250)
    if df is None:
        return None

    df["fast"] = df["close"].ewm(span=FAST_EMA, adjust=False).mean()
    df["slow"] = df["close"].ewm(span=SLOW_EMA, adjust=False).mean()

    # Closed candle crossover
    if not (df["fast"].iloc[-2] < df["slow"].iloc[-2] and df["fast"].iloc[-1] > df["slow"].iloc[-1]) and \
       not (df["fast"].iloc[-2] > df["slow"].iloc[-2] and df["fast"].iloc[-1] < df["slow"].iloc[-1]):
        return None

    # Higher timeframe trend
    df_h4 = get_rates(symbol, TREND_TIMEFRAME, 120)
    if df_h4 is None:
        return None
    df_h4["ema"] = df_h4["close"].ewm(span=TREND_EMA, adjust=False).mean()
    price = df_h4["close"].iloc[-1]
    ema = df_h4["ema"].iloc[-1]

    if df["fast"].iloc[-1] > df["slow"].iloc[-1] and price > ema:
        return "buy"
    if df["fast"].iloc[-1] < df["slow"].iloc[-1] and price < ema:
        return "sell"
    return None

def get_position_size(symbol, sl_distance):
    acc = mt5.account_info()
    if acc is None:
        return 0.01
    risk_money = acc.equity * (RISK_PERCENT / 100.0)
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_tick_value == 0:
        return info.volume_min if info else 0.01

    money_per_lot = (sl_distance / info.trade_tick_size) * info.trade_tick_value
    if money_per_lot <= 0:
        return info.volume_min

    lots = risk_money / money_per_lot
    step = info.volume_step
    lots = max(info.volume_min, min(info.volume_max, lots))
    lots = round(lots / step) * step
    return round(lots, 2)

def get_filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    if info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    if info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def log_trade(symbol, order_type, volume, price, sl, tp, ticket, comment=""):
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), symbol, order_type, volume,
            price, sl, tp, ticket, comment
        ])

def open_trade(symbol, signal):
    info = mt5.symbol_info(symbol)
    if info is None:
        return
    if not info.visible:
        mt5.symbol_select(symbol, True)

    df = get_rates(symbol, TIMEFRAME, 60)
    if df is None:
        return
    atr = calc_atr(df, ATR_PERIOD)
    if atr <= 0:
        return

    tick = mt5.symbol_info_tick(symbol)
    digits = info.digits

    if signal == "buy":
        price = tick.ask
        sl = price - atr * ATR_SL_MULTIPLIER
        tp = price + atr * ATR_TP_MULTIPLIER
        order_type = mt5.ORDER_TYPE_BUY
        type_str = "BUY"
    else:
        price = tick.bid
        sl = price + atr * ATR_SL_MULTIPLIER
        tp = price - atr * ATR_TP_MULTIPLIER
        order_type = mt5.ORDER_TYPE_SELL
        type_str = "SELL"

    price = round(price, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    sl_dist = abs(price - sl)
    volume = get_position_size(symbol, sl_dist)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 40,
        "magic": MAGIC,
        "comment": "AdvEMA",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"OPEN {type_str} {symbol} | Vol:{volume} SL:{sl} TP:{tp} Ticket:{result.order}")
        log_trade(symbol, type_str, volume, price, sl, tp, result.order)
        send_telegram(f"✅ {type_str} {symbol}\nVol: {volume}\nSL: {sl}\nTP: {tp}")
    else:
        logger.error(f"Order fail {symbol}: {result.retcode} {result.comment}")

def manage_positions():
    """Trailing stop + Break-even"""
    positions = mt5.positions_get()
    if positions is None:
        return

    for pos in positions:
        if pos.magic != MAGIC:
            continue

        symbol = pos.symbol
        info = mt5.symbol_info(symbol)
        if info is None:
            continue

        df = get_rates(symbol, TIMEFRAME, 50)
        if df is None:
            continue
        atr = calc_atr(df, ATR_PERIOD)
        if atr <= 0:
            continue

        tick = mt5.symbol_info_tick(symbol)
        digits = info.digits
        point = info.point

        # Current profit in price
        if pos.type == mt5.ORDER_TYPE_BUY:
            current_price = tick.bid
            profit_dist = current_price - pos.price_open
            new_sl = None

            # Break-even
            if profit_dist >= (pos.price_open - pos.sl) * BREAKEVEN_TRIGGER_R and pos.sl < pos.price_open:
                new_sl = pos.price_open + 2 * point   # tiny buffer

            # Trailing
            if profit_dist >= atr * TRAIL_START_ATR:
                trail_sl = current_price - atr * TRAIL_STEP_ATR
                if trail_sl > pos.sl:
                    new_sl = max(new_sl or 0, trail_sl)

        else:  # SELL
            current_price = tick.ask
            profit_dist = pos.price_open - current_price
            new_sl = None

            if profit_dist >= (pos.sl - pos.price_open) * BREAKEVEN_TRIGGER_R and pos.sl > pos.price_open:
                new_sl = pos.price_open - 2 * point

            if profit_dist >= atr * TRAIL_START_ATR:
                trail_sl = current_price + atr * TRAIL_STEP_ATR
                if trail_sl < pos.sl or pos.sl == 0:
                    new_sl = min(new_sl or 999999, trail_sl) if new_sl else trail_sl

        if new_sl is not None:
            new_sl = round(new_sl, digits)
            if abs(new_sl - pos.sl) > point:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "symbol": symbol,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Modified SL {symbol} → {new_sl} (ticket {pos.ticket})")

def count_my_positions():
    positions = mt5.positions_get()
    if positions is None:
        return 0
    return sum(1 for p in positions if p.magic == MAGIC)

def daily_loss_exceeded():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    deals = mt5.history_deals_get(today, datetime.now())
    if not deals:
        return False
    profit = sum(d.profit for d in deals if d.magic == MAGIC)
    acc = mt5.account_info()
    if acc is None or acc.balance <= 0:
        return False
    loss_pct = (profit / acc.balance) * 100
    if loss_pct < -MAX_DAILY_LOSS_PERCENT:
        logger.warning(f"Daily loss limit hit: {loss_pct:.2f}%")
        send_telegram(f"⚠️ Daily loss limit reached ({loss_pct:.2f}%). Paused.")
        return True
    return False

def is_trading_hours():
    if TRADING_HOURS is None:
        return True
    h = datetime.now().hour
    return TRADING_HOURS[0] <= h < TRADING_HOURS[1]

def print_stats():
    """Simple performance summary from log"""
    if not TRADE_LOG.exists():
        return
    try:
        df = pd.read_csv(TRADE_LOG)
        logger.info(f"Total trades logged: {len(df)}")
    except:
        pass

def run():
    if not connect():
        return

    logger.info("=== Advanced EMA Bot Started ===")
    print_stats()

    try:
        while True:
            # Reconnect if needed
            if not mt5.terminal_info() or not mt5.terminal_info().connected:
                logger.warning("Disconnected → reconnecting...")
                mt5.shutdown()
                time.sleep(5)
                if not connect():
                    time.sleep(30)
                    continue

            # Manage existing positions first
            manage_positions()

            if daily_loss_exceeded():
                time.sleep(180)
                continue

            if not is_trading_hours():
                time.sleep(60)
                continue

            if count_my_positions() >= MAX_OPEN_TRADES:
                time.sleep(CHECK_SECONDS)
                continue

            for symbol in SYMBOLS:
                # Skip if already have position
                positions = mt5.positions_get(symbol=symbol)
                if positions and any(p.magic == MAGIC for p in positions):
                    continue

                signal = get_signal(symbol)
                if signal:
                    open_trade(symbol, signal)
                    time.sleep(1.5)

            time.sleep(CHECK_SECONDS)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        send_telegram("🛑 Bot stopped manually")
    except Exception as e:
        logger.exception(f"Critical error: {e}")
        send_telegram(f"❌ Bot crashed: {e}")
    finally:
        mt5.shutdown()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    run()