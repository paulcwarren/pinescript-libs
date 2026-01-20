import yfinance as yf
import numpy as np
import math
import json
from datetime import date
from scipy.stats import norm
import logging

# -----------------------------
# CONFIG
# -----------------------------
INDEXES = ["SPY", "QQQ", "IWM", "DIA"]

# Your sector/ETF universe
ETFS = ["BLOK", "IGV", "CLOU", "MAGS", "QTUM", "URA", "UFO", "ROBO", "OIH",
        "XLK", "XLF", "XLY", "XLI", "XLE", "XLC", "XLV", "XLU", "XLRE",
        "XHB", "XBI", "XLP", "SOXX", "XME", "XRT"]

TOP_HOLDINGS_COUNT = 10  # number of top holdings to fetch dynamically

# Ticker mapping for Yahoo Finance
TICKER_MAP = {
    "BRK.B": "BRK-B",
}

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# Black-Scholes Gamma
# -----------------------------
def gamma(S, K, sigma, T, r=0.0):
    if T == 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# -----------------------------
# Wall Calculation
# -----------------------------
def calculate_walls(chain_calls, chain_puts, spot):
    strikes = sorted(set(chain_calls['strike']).union(set(chain_puts['strike'])))
    gex_per_strike = {}

    for K in strikes:
        call = chain_calls[chain_calls['strike'] == K]
        put  = chain_puts[chain_puts['strike'] == K]

        # handle NaN or missing values
        call_oi = int(call['openInterest'].values[0]) if len(call) and not np.isnan(call['openInterest'].values[0]) else 0
        put_oi  = int(put['openInterest'].values[0])  if len(put)  and not np.isnan(put['openInterest'].values[0])  else 0
        call_iv = float(call['impliedVolatility'].values[0]) if len(call) and not np.isnan(call['impliedVolatility'].values[0]) else 0.2
        put_iv  = float(put['impliedVolatility'].values[0])  if len(put)  and not np.isnan(put['impliedVolatility'].values[0])  else 0.2

        T = 1/252  # 1 day ~ 1/252 year
        call_gex = -call_oi * gamma(spot, K, call_iv, T) * spot**2
        put_gex  = +put_oi  * gamma(spot, K, put_iv, T) * spot**2
        gex_per_strike[K] = call_gex + put_gex

    # put wall = max positive GEX below spot
    put_candidates = [k for k, g in gex_per_strike.items() if k <= spot]
    put_wall = max(put_candidates, key=lambda k: gex_per_strike[k], default=None)

    # call wall = max negative GEX above spot
    call_candidates = [k for k, g in gex_per_strike.items() if k >= spot]
    call_wall = min(call_candidates, key=lambda k: gex_per_strike[k], default=None)

    # gamma flip = strike where cumulative GEX crosses zero
    cumulative = 0
    gamma_flip = None
    for k in sorted(strikes):
        cumulative += gex_per_strike[k]
        if cumulative >= 0:
            gamma_flip = k
            break

    return put_wall, call_wall, gamma_flip

# -----------------------------
# Process ticker with safeguards
# -----------------------------
def process_ticker(ticker):
    yf_ticker = TICKER_MAP.get(ticker, ticker)
    try:
        logging.info(f"Processing {ticker}...")
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            logging.warning(f"No price data for {ticker}")
            return None
        spot = hist['Close'].iloc[-1]

        expirations = stock.options
        if not expirations:
            logging.warning(f"No option chain for {ticker}")
            return None
        expiry = expirations[0]

        chain = stock.option_chain(expiry)
        put_wall, call_wall, gamma_flip = calculate_walls(chain.calls, chain.puts, spot)

        logging.info(f"{ticker} processed: spot={spot}, putWall={put_wall}, callWall={call_wall}, gammaFlip={gamma_flip}")
        return {
            "spot": round(float(spot), 2),
            "putWall": put_wall,
            "callWall": call_wall,
            "gammaFlip": gamma_flip,
            "expiry": expiry
        }
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return None

# -----------------------------
# Fetch ETF top holdings dynamically
# -----------------------------
def get_top_holdings(etf, n=10):
    yf_ticker = TICKER_MAP.get(etf, etf)
    try:
        t = yf.Ticker(yf_ticker)
        holdings = t.funds_data.top_holdings
        if holdings is None or holdings.empty:
            logging.warning(f"No holdings found for {etf}")
            return []
        top_symbols = holdings.index.tolist()[:n]
        logging.info(f"Top holdings for {etf}: {top_symbols}")
        return top_symbols
    except Exception as e:
        logging.error(f"Error fetching holdings for {etf}: {e}")
        return []

# -----------------------------
# Main loop
# -----------------------------
walls_dict = {}
today_str = date.today().isoformat()

# 1) INDEXES
for idx in INDEXES:
    result = process_ticker(idx)
    if result:
        walls_dict[idx] = result

# 2) ETFs and their top holdings
for etf in ETFS:
    # ETF itself
    result = process_ticker(etf)
    if result:
        walls_dict[etf] = result

    # Top holdings
    holdings = get_top_holdings(etf, TOP_HOLDINGS_COUNT)
    for h in holdings:
        if h in walls_dict:  # avoid duplicates
            continue
        res_h = process_ticker(h)
        if res_h:
            walls_dict[h] = res_h

# -----------------------------
# Write JS file for GitHub Pages
# -----------------------------
js_text = "window.WALLS = " + json.dumps(walls_dict, indent=2) + ";"

with open("options/walls.js", "w") as f:
    f.write(js_text)

logging.info("walls.js generated successfully!")
