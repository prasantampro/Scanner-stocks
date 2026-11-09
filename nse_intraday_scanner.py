"""
NSE intraday scanner using Yahoo Finance (yfinance).

Setup:   pip install yfinance pandas numpy
Run:     python nse_intraday_scanner.py            (loops every 5 min in market hours)
         python nse_intraday_scanner.py --once     (single scan, works any time)

Signals (5-minute candles):
  BULLISH : price > VWAP, price > opening-range high (first 15 min),
            relative volume >= 1.5x, RSI(14) between 55 and 80
  BEARISH : mirror image (price < VWAP, < opening-range low, RSI 20-45)

Notes:
  - Yahoo data for NSE is delayed (typically ~15 min) and unofficial.
    Use it for learning/screening, not for live order execution.
  - 5m interval data is available for the last ~60 days; 1m only for ~7 days.
"""
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- Config ----------------
SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI",
    "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "NTPC", "ONGC", "ADANIENT",
    "WIPRO", "HCLTECH", "M&M", "ASIANPAINT", "TITAN", "POWERGRID", "COALINDIA",
    "JSWSTEEL", "ULTRACEMCO", "NESTLEIND",
]
INTERVAL = "5m"
PERIOD = "5d"               # need previous days for prev close & volume baseline
OPENING_RANGE_BARS = 3      # first 3 x 5m = 15 min
RVOL_MIN = 1.5
SCAN_EVERY_SECONDS = 300
IST = "Asia/Kolkata"


# ---------------- Indicators ----------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def vwap(day: pd.DataFrame) -> pd.Series:
    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    return (typical * day["Volume"]).cumsum() / day["Volume"].cumsum()


def relative_volume(df: pd.DataFrame, today_date, n_bars: int) -> float:
    """Today's cumulative volume so far vs. average of previous days at the same time."""
    today_vol = df[df.index.date == today_date]["Volume"].sum()
    past = []
    for d, g in df[df.index.date < today_date].groupby(df[df.index.date < today_date].index.date):
        past.append(g["Volume"].iloc[:n_bars].sum())
    avg = np.mean(past) if past else np.nan
    return today_vol / avg if avg and avg > 0 else np.nan


# ---------------- Scanner ----------------
def analyze(df: pd.DataFrame):
    df = df.dropna(subset=["Close"])
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    last_date = df.index[-1].date()
    today = df[df.index.date == last_date]
    prev = df[df.index.date < last_date]
    if len(today) < OPENING_RANGE_BARS or prev.empty:
        return None

    prev_close = prev["Close"].iloc[-1]
    ltp = today["Close"].iloc[-1]
    or_high = today["High"].iloc[:OPENING_RANGE_BARS].max()
    or_low = today["Low"].iloc[:OPENING_RANGE_BARS].min()
    v = vwap(today).iloc[-1]
    r = rsi(df["Close"]).iloc[-1]
    rvol = relative_volume(df, last_date, len(today))

    signal = ""
    if ltp > v and ltp > or_high and rvol >= RVOL_MIN and 55 <= r <= 80:
        signal = "BULLISH"
    elif ltp < v and ltp < or_low and rvol >= RVOL_MIN and 20 <= r <= 45:
        signal = "BEARISH"

    return {
        "LTP": round(ltp, 2),
        "Chg%": round((ltp / prev_close - 1) * 100, 2),
        "Gap%": round((today["Open"].iloc[0] / prev_close - 1) * 100, 2),
        "VWAP": round(v, 2),
        "OR_High": round(or_high, 2),
        "OR_Low": round(or_low, 2),
        "RSI": round(r, 1),
        "RVOL": round(rvol, 2) if pd.notna(rvol) else np.nan,
        "Signal": signal,
        "BarTime": today.index[-1].strftime("%H:%M"),
    }


def run_scan() -> pd.DataFrame:
    tickers = [f"{s}.NS" for s in SYMBOLS]
    data = yf.download(
        tickers, period=PERIOD, interval=INTERVAL, group_by="ticker",
        auto_adjust=False, threads=True, progress=False,
    )
    rows = []
    for s, t in zip(SYMBOLS, tickers):
        try:
            res = analyze(data[t].copy())
        except Exception as e:  # one bad ticker shouldn't kill the scan
            print(f"  skip {s}: {e}")
            continue
        if res:
            res["Symbol"] = s
            rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cols = ["Symbol", "Signal", "LTP", "Chg%", "Gap%", "VWAP", "OR_High", "OR_Low", "RSI", "RVOL", "BarTime"]
    return out[cols].sort_values(["Signal", "RVOL"], ascending=[False, False]).reset_index(drop=True)


def market_open_now() -> bool:
    now = pd.Timestamp.now(tz=IST)
    if now.weekday() >= 5:
        return False
    return dt.time(9, 15) <= now.time() <= dt.time(15, 30)


if __name__ == "__main__":
    once = "--once" in sys.argv
    while True:
        if once or market_open_now():
            stamp = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n=== Scan @ {stamp} IST ===")
            result = run_scan()
            if result.empty:
                print("No data returned.")
            else:
                hits = result[result["Signal"] != ""]
                print(hits.to_string(index=False) if not hits.empty else "No signals this round.")
                result.to_csv("scan_latest.csv", index=False)
            if once:
                break
        else:
            print("Market closed; waiting...")
        time.sleep(SCAN_EVERY_SECONDS)
