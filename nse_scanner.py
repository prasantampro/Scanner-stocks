"""
Simple NSE stock scanner (end-of-day data via Yahoo Finance).

Setup:
    pip install yfinance pandas

Run:
    python nse_scanner.py
"""
import pandas as pd
import yfinance as yf

# NSE symbols use the ".NS" suffix on Yahoo Finance.
# Replace with your own list, or load the full list from NSE's
# "EQUITY_L.csv" (https://www.nseindia.com/market-data/securities-available-for-trading)
SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN",
    "ITC", "LT", "BHARTIARTL", "AXISBANK", "KOTAKBANK", "TATAMOTORS",
    "SUNPHARMA", "MARUTI", "TITAN", "ADANIENT", "WIPRO", "ONGC",
]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def analyse(df: pd.DataFrame) -> dict | None:
    if len(df) < 60:
        return None
    close, high, vol = df["Close"], df["High"], df["Volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = rsi(close)
    avg_vol20 = vol.rolling(20).mean()
    high20_prev = high.rolling(20).max().shift(1)  # prior 20-day high

    last = df.index[-1]
    return {
        "close": round(close[last], 2),
        "chg_%": round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
        "rsi": round(rsi14.iloc[-1], 1),
        "above_sma50": close.iloc[-1] > sma50.iloc[-1],
        "sma20_gt_sma50": sma20.iloc[-1] > sma50.iloc[-1],
        "vol_x_avg": round(vol.iloc[-1] / avg_vol20.iloc[-1], 2),
        "breakout_20d": close.iloc[-1] > high20_prev.iloc[-1],
    }


def main():
    tickers = [s + ".NS" for s in SYMBOLS]
    data = yf.download(
        tickers, period="1y", interval="1d",
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )

    rows = []
    for sym in SYMBOLS:
        try:
            df = data[sym + ".NS"].dropna()
            res = analyse(df)
            if res:
                rows.append({"symbol": sym, **res})
        except Exception as e:
            print(f"skip {sym}: {e}")

    table = pd.DataFrame(rows)

    # ---- Define your scans here ----
    breakout = table[table["breakout_20d"] & (table["vol_x_avg"] >= 1.5)]
    oversold = table[(table["rsi"] < 30)]
    trend = table[table["above_sma50"] & table["sma20_gt_sma50"] & (table["rsi"].between(50, 70))]

    pd.set_option("display.width", 160)
    print("\n== Breakout with volume (20d high, vol >= 1.5x avg) ==")
    print(breakout.to_string(index=False) if not breakout.empty else "none")
    print("\n== Oversold (RSI < 30) ==")
    print(oversold.to_string(index=False) if not oversold.empty else "none")
    print("\n== Uptrend (above SMA50, SMA20 > SMA50, RSI 50-70) ==")
    print(trend.to_string(index=False) if not trend.empty else "none")

    table.to_csv("scan_results.csv", index=False)
    print("\nFull results saved to scan_results.csv")


if __name__ == "__main__":
    main()
