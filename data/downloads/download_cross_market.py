#!/usr/bin/env python3
"""Download cross-market daily data from Yahoo Finance for BTC alpha features.

Fetches SPY, QQQ, VIX, TLT, USO, COIN daily closes and computes
derived features (returns, z-scores, extremes).

Output: data_files/cross_market_daily.csv
  Columns: date, spy_ret_1d, qqq_ret_1d, spy_ret_5d, vix_level,
           tlt_ret_5d, uso_ret_5d, coin_ret_1d, spy_extreme

Usage:
    python3 -m data.downloads.download_cross_market
    python3 -m data.downloads.download_cross_market --days 30  # last 30 days only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("cross_market")

TICKERS = {
    # Macro (V1)
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "^VIX": "VIX",
    "TLT": "US Treasury 20Y",
    "USO": "Oil ETF",
    "COIN": "Coinbase",
    # Macro V2
    "IEF": "US Treasury 7-10Y",
    "XLF": "Financial Sector",
    "EEM": "Emerging Markets",
    "GLD": "Gold",
    "^TNX": "10Y Treasury Yield",
    # Crypto ETFs V3 (IC -0.08 to -0.10)
    "IBIT": "iShares Bitcoin ETF",
    "GBTC": "Grayscale Bitcoin Trust",
    "ETHE": "Grayscale Ethereum Trust",
    "BITO": "ProShares Bitcoin Futures",
    # V4: Additional crypto ETFs + miners
    "ETHA": "iShares Ethereum ETF",
    "BITX": "2x Bitcoin ETF",
    "BITI": "Short Bitcoin ETF",
    "MARA": "Marathon Digital",
    "RIOT": "Riot Platforms",
    # Trad-fi risk/macro (V12+)
    "HYG": "High Yield Corporate Bond",
    "IWM": "Russell 2000 Small Cap",
    "XLK": "Technology Sector",
    "FXI": "China Large Cap",
    # BTC spot for gold-BTC divergence
    "BTC-USD": "Bitcoin USD",
}

OUTPUT_PATH = Path("data_files/cross_market_daily.csv")


def fetch_yahoo(ticker: str, range_str: str = "5y") -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{ticker}?range={range_str}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read())
    result = raw["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s"),
        "close": q["close"],
    }).dropna(subset=["close"])
    df["date"] = df["date"].dt.date
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.set_index("date")


def _merge_fred_macro(out: pd.DataFrame) -> None:
    """Merge FRED macro data (M2 + yield curve 2s10s) into cross-market features.

    Reads pre-downloaded CSVs from data_files/macro/. If files don't exist,
    attempts a fresh download via download_fred_macro.
    """
    macro_dir = Path("data_files/macro")

    # ── M2 Money Supply ──
    m2_path = macro_dir / "m2_money_supply.csv"
    if not m2_path.exists():
        try:
            from data.downloads.download_fred_macro import download_m2
            download_m2()
        except Exception as e:
            log.warning("M2 auto-download failed: %s", e)

    if m2_path.exists():
        try:
            m2 = pd.read_csv(m2_path, parse_dates=["date"])
            m2["date"] = m2["date"].dt.date
            m2 = m2.set_index("date")
            # Align to out's index (daily dates)
            for col in ["m2_yoy_change", "m2_3m_change", "m2_mom_change"]:
                if col in m2.columns:
                    aligned = m2[col].reindex(out.index).ffill()
                    out[col] = aligned
            log.info("M2 features merged: %d rows", len(m2))
        except Exception as e:
            log.warning("M2 merge failed: %s", e)

    # ── Yield Curve 2s10s ──
    yc_path = macro_dir / "yield_curve_2s10s.csv"
    if not yc_path.exists():
        try:
            from data.downloads.download_fred_macro import download_yield_curve_2s10s
            download_yield_curve_2s10s()
        except Exception as e:
            log.warning("Yield curve auto-download failed: %s", e)

    if yc_path.exists():
        try:
            yc = pd.read_csv(yc_path, parse_dates=["date"])
            yc["date"] = yc["date"].dt.date
            yc = yc.set_index("date")
            # Level (spread in %)
            aligned = yc["value"].reindex(out.index).ffill()
            out["yield_curve_2s10s"] = aligned
            # 5-day change in spread
            out["yield_curve_2s10s_chg_5d"] = aligned.diff(5)
            # Inversion flag: negative spread = inverted yield curve
            out["yield_curve_inverted"] = (aligned < 0).astype(float)
            log.info("Yield curve 2s10s features merged: %d rows", len(yc))
        except Exception as e:
            log.warning("Yield curve merge failed: %s", e)


def build_cross_market_features() -> pd.DataFrame:
    """Download all tickers and compute cross-market features."""
    data = {}
    for ticker, name in TICKERS.items():
        try:
            df = fetch_yahoo(ticker)
            data[ticker] = df["close"]
            log.info("%s (%s): %d days", ticker, name, len(df))
        except Exception as e:
            log.warning("%s failed: %s", ticker, e)

    if "SPY" not in data:
        log.error("SPY download failed — cannot compute features")
        sys.exit(1)

    # Align all to SPY dates
    combined = pd.DataFrame(data)
    combined = combined.dropna(subset=["SPY"])

    out = pd.DataFrame(index=combined.index)
    out.index.name = "date"

    # Returns
    out["spy_ret_1d"] = combined["SPY"].pct_change()
    out["qqq_ret_1d"] = combined["QQQ"].pct_change() if "QQQ" in combined else None
    out["spy_ret_5d"] = combined["SPY"].pct_change(5)
    out["coin_ret_1d"] = combined["COIN"].pct_change() if "COIN" in combined else None

    # 5d returns
    out["tlt_ret_5d"] = combined["TLT"].pct_change(5) if "TLT" in combined else None
    out["uso_ret_5d"] = combined["USO"].pct_change(5) if "USO" in combined else None

    # VIX level (not return — absolute level matters)
    out["vix_level"] = combined["^VIX"] if "^VIX" in combined else None

    # V2: Macro proxy features (from Yahoo ETFs, replacing FRED)
    # 10Y Treasury yield level
    if "^TNX" in combined:
        out["treasury_10y"] = combined["^TNX"]
        out["treasury_10y_chg_5d"] = combined["^TNX"].diff(5)

    # Yield curve proxy: TLT (long bonds) vs IEF (medium bonds)
    if "TLT" in combined and "IEF" in combined:
        out["yield_curve_proxy"] = combined["TLT"].pct_change(20) - combined["IEF"].pct_change(20)

    # Financial sector (rate sensitivity)
    if "XLF" in combined:
        out["xlf_ret_5d"] = combined["XLF"].pct_change(5)

    # Emerging markets (risk appetite)
    if "EEM" in combined:
        out["eem_ret_5d"] = combined["EEM"].pct_change(5)

    # Gold (safe haven)
    if "GLD" in combined:
        out["gld_ret_5d"] = combined["GLD"].pct_change(5)
        out["gld_ret_1d"] = combined["GLD"].pct_change()

    # Gold-BTC divergence features
    if "GLD" in combined and "BTC-USD" in combined:
        gld_ret = combined["GLD"].pct_change()
        btc_ret = combined["BTC-USD"].pct_change()
        # 30-day rolling correlation between GLD and BTC daily returns
        out["gold_btc_corr_30d"] = gld_ret.rolling(30).corr(btc_ret)
        # 5-day return spread: gold outperforming = positive
        out["gold_btc_return_spread_5d"] = (
            combined["GLD"].pct_change(5) - combined["BTC-USD"].pct_change(5)
        )

    # SPY extreme flag: |ret| > 2%
    if "spy_ret_1d" in out:
        out["spy_extreme"] = 0.0
        out.loc[out["spy_ret_1d"] > 0.02, "spy_extreme"] = 1.0
        out.loc[out["spy_ret_1d"] < -0.02, "spy_extreme"] = -1.0

    # V3: Crypto ETF features (strongest signals: IC -0.08 to -0.10)
    if "ETHE" in combined:
        out["ethe_ret_1d"] = combined["ETHE"].pct_change()
    if "GBTC" in combined:
        out["gbtc_ret_1d"] = combined["GBTC"].pct_change()
    if "IBIT" in combined:
        out["ibit_ret_1d"] = combined["IBIT"].pct_change()
    if "BITO" in combined:
        out["bito_ret_1d"] = combined["BITO"].pct_change()

    # GBTC premium/discount (GBTC price vs BTC spot proxy via IBIT)
    if "GBTC" in combined and "IBIT" in combined:
        ratio = combined["GBTC"] / combined["IBIT"]
        ratio_ma = ratio.rolling(20).mean()
        out["gbtc_premium_dev"] = (ratio / ratio_ma - 1) if ratio_ma is not None else None

    # V4: Additional crypto ETFs + miners
    if "ETHA" in combined:
        out["etha_ret_1d"] = combined["ETHA"].pct_change()
    if "BITX" in combined:
        out["bitx_ret_1d"] = combined["BITX"].pct_change()  # 2x leverage amplifies sentiment
    if "BITI" in combined:
        out["biti_ret_1d"] = combined["BITI"].pct_change()   # inverse = short sentiment
    if "MARA" in combined:
        out["mara_ret_1d"] = combined["MARA"].pct_change()   # miner behavior
    if "RIOT" in combined:
        out["riot_ret_1d"] = combined["RIOT"].pct_change()

    # V12+: Trad-fi risk/macro factors
    # HYG: high yield corporate bond — risk appetite indicator
    if "HYG" in combined:
        out["hyg_ret_1d"] = combined["HYG"].pct_change()
        out["hyg_ret_5d"] = combined["HYG"].pct_change(5)
        # Credit spread proxy: TLT (safe) vs HYG (risky) divergence
        if "TLT" in combined:
            out["credit_spread_chg"] = combined["TLT"].pct_change(5) - combined["HYG"].pct_change(5)

    # IWM: Russell 2000 — small cap risk appetite
    if "IWM" in combined:
        out["iwm_ret_1d"] = combined["IWM"].pct_change()
        # Risk-on/off: IWM vs SPY (small cap underperform = risk-off)
        if "SPY" in combined:
            out["risk_appetite"] = combined["IWM"].pct_change(5) - combined["SPY"].pct_change(5)

    # XLK: Technology sector
    if "XLK" in combined:
        out["xlk_ret_1d"] = combined["XLK"].pct_change()

    # VIX change rate (more useful than level)
    if "^VIX" in combined:
        out["vix_chg_1d"] = combined["^VIX"].pct_change()
        out["vix_chg_5d"] = combined["^VIX"].pct_change(5)

    # FXI: China large cap — Asia risk sentiment
    if "FXI" in combined:
        out["fxi_ret_1d"] = combined["FXI"].pct_change()

    # ── FRED macro: M2 money supply + yield curve 2s10s ──
    _merge_fred_macro(out)

    return out.dropna(subset=["spy_ret_1d"])


def build_etf_volume() -> None:
    """Download ETF dollar volume data and save to etf_volume_daily.csv."""
    etf_tickers = ["IBIT", "GBTC", "FBTC", "ETHA", "ETHE", "BITO"]
    all_data = {}
    for ticker in etf_tickers:
        try:
            df = fetch_yahoo(ticker)
            if len(df) > 10:
                all_data[ticker] = df
                log.info("ETF vol %s: %d days", ticker, len(df))
        except Exception as e:
            log.debug("ETF vol %s failed: %s", ticker, e)

    if not all_data:
        log.warning("No ETF volume data downloaded")
        return

    # Use longest ETF as date index
    longest = max(all_data, key=lambda t: len(all_data[t]))
    dates = all_data[longest].index
    out = pd.DataFrame({"date": dates})

    # Aggregate BTC ETF dollar volume
    agg = pd.Series(0.0, index=dates)
    for t in ["IBIT", "GBTC", "FBTC", "BITO"]:
        if t in all_data:
            dvol = all_data[t]["close"] * all_data[t]["volume"]
            agg = agg.add(dvol.reindex(dates, fill_value=0), fill_value=0)
    out["btc_etf_dollar_vol"] = agg.values

    # Per-ETF dollar volume
    for t in ["IBIT", "GBTC", "ETHA"]:
        if t in all_data:
            dvol = all_data[t]["close"] * all_data[t]["volume"]
            out[f"{t.lower()}_dollar_vol"] = dvol.reindex(dates, fill_value=0).values

    etf_vol_path = Path("data_files/etf_volume_daily.csv")
    out.to_csv(etf_vol_path, index=False)
    log.info("ETF volume: %d days → %s", len(out), etf_vol_path)


def main():
    parser = argparse.ArgumentParser(description="Download cross-market data")
    parser.add_argument("--days", type=int, default=None,
                        help="Only keep last N days")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    features = build_cross_market_features()

    if args.days:
        features = features.tail(args.days)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH)
    log.info("Saved %d rows to %s", len(features), OUTPUT_PATH)

    # Download FRED macro data (M2 + yield curve) — saved as individual CSVs
    try:
        from data.downloads.download_fred_macro import download_m2, download_yield_curve_2s10s
        download_m2()
        download_yield_curve_2s10s()
    except Exception as e:
        log.warning("FRED macro download failed (non-critical): %s", e)

    # Also update ETF volume data (optional — some ETFs may lack volume)
    try:
        build_etf_volume()
    except Exception as e:
        log.warning("ETF volume update failed (non-critical): %s", e)

    # Save individual ETF daily files to data_files/macro/ for alpha_builder
    # cross_market_source (CsvCursor reads SPY_daily.csv, TLT_daily.csv, etc.)
    macro_dir = Path("data_files/macro")
    macro_dir.mkdir(parents=True, exist_ok=True)
    etf_map = {
        "SPY": "SPY_daily.csv",
        "TLT": "TLT_daily.csv",
        "^VIX": "VIX_daily.csv",
        "USO": "USO_daily.csv",
        "GLD": "GLD_daily.csv",
        "COIN": "COIN_daily.csv",
        "UUP": "UUP_daily.csv",
    }
    for ticker, name in TICKERS.items():
        fname = etf_map.get(ticker)
        if fname is None:
            continue
        try:
            df = fetch_yahoo(ticker)
            if len(df) > 0:
                df.to_csv(macro_dir / fname)
                log.info("Saved %s: %d rows", fname, len(df))
        except Exception as e:
            log.debug("Failed to save %s: %s", fname, e)

    # Summary
    print(f"\nCross-market features: {len(features)} days")
    print(f"Columns: {list(features.columns)}")
    print(f"Range: {features.index[0]} → {features.index[-1]}")


if __name__ == "__main__":
    main()
