import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
import json
from datetime import date, datetime, timedelta
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
INDEXES = ["SPY", "QQQ", "IWM", "DIA"]

ETFS = [
    "BLOK", "IGV", "CLOU", "MAGS", "QTUM", "URA", "UFO", "ROBO", "OIH",
    "XLK", "XLF", "XLY", "XLI", "XLE", "XLC", "XLV", "XLU", "XLRE",
    "XHB", "XBI", "XLP", "SOXX", "XME", "XRT"
]

TOP_HOLDINGS_COUNT = 10  # Number of top holdings to fetch per ETF
EXPIRATION_LOOKAHEAD = 3 # Aggregate OI/Gamma across next 3 expirations

# Ticker mapping for Yahoo Finance (Dot to Dash conversion)
TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# MATH: Black-Scholes Gamma
# -----------------------------
def calc_gamma(S, K, sigma, T, r=0.015):
    """
    Calculates Gamma for a given contract.
    T is clamped to a minimum of 0.001 (approx 4 hours) to avoid division by zero errors on expiration day.
    """
    T = np.maximum(T, 0.001)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma_val

# -----------------------------
# CORE LOGIC: Process Option Chain
# -----------------------------
def get_gex_and_walls(ticker):
    # Normalize ticker symbol (e.g. BRK.B -> BRK-B)
    yf_ticker = TICKER_MAP.get(ticker, ticker)
    stock = yf.Ticker(yf_ticker)
    
    try:
        # 1. Get Spot Price
        # We use history(period="1d") because it's generally more reliable than info['regularMarketPrice']
        hist = stock.history(period="1d")
        if hist.empty:
            logging.warning(f"No price data found for {ticker}")
            return None
        spot = hist['Close'].iloc[-1]
        
        # 2. Get Expirations
        exps = stock.options
        if not exps:
            logging.warning(f"No option chain found for {ticker}")
            return None
            
        # Limit to the next N expirations to simulate "Total Gamma" without over-fetching
        target_exps = exps[:EXPIRATION_LOOKAHEAD]
        
        all_calls = []
        all_puts = []

        # 3. Iterate Expirations and Collect Chains
        for exp_date_str in target_exps:
            try:
                chain = stock.option_chain(exp_date_str)
                
                # Calculate Time to Expiry (T) in years
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                today = date.today()
                days_to_expiry = (exp_date - today).days
                
                # If 0DTE (or negative due to timezone diffs), clamp to 0.5 days
                T = max(days_to_expiry, 0.5) / 365.0
                
                # Calls
                calls = chain.calls.copy()
                calls['type'] = 'call'
                calls['T'] = T
                all_calls.append(calls)
                
                # Puts
                puts = chain.puts.copy()
                puts['type'] = 'put'
                puts['T'] = T
                all_puts.append(puts)
            except Exception as e:
                # Sometimes specific expirations fail in YF, skip them but continue processing
                continue

        if not all_calls or not all_puts:
            return None

        # Concatenate all chains into single DataFrames
        df_calls = pd.concat(all_calls)
        df_puts = pd.concat(all_puts)
        
        # 4. Data Cleaning
        # Fill missing Implied Volatility with a placeholder (0.2 = 20%) to allow GEX calc
        df_calls['impliedVolatility'] = df_calls['impliedVolatility'].replace(0, np.nan).fillna(0.2)
        df_puts['impliedVolatility'] = df_puts['impliedVolatility'].replace(0, np.nan).fillna(0.2)
        
        # 5. Calculate Gamma (Vectorized)
        # Call GEX = Negative (Dealer Short Gamma)
        # Put GEX = Positive (Dealer Long Gamma)
        df_calls['gamma'] = calc_gamma(spot, df_calls['strike'], df_calls['impliedVolatility'], df_calls['T'])
        df_calls['GEX'] = df_calls['gamma'] * df_calls['openInterest'] * (spot**2) * -1 
        
        df_puts['gamma'] = calc_gamma(spot, df_puts['strike'], df_puts['impliedVolatility'], df_puts['T'])
        df_puts['GEX'] = df_puts['gamma'] * df_puts['openInterest'] * (spot**2) 
        
        # 6. Aggregate by Strike
        call_stats = df_calls.groupby('strike')[['openInterest', 'GEX']].sum()
        put_stats = df_puts.groupby('strike')[['openInterest', 'GEX']].sum()
        
        total_df = pd.DataFrame(index=sorted(set(call_stats.index) | set(put_stats.index)))
        total_df['call_OI'] = call_stats['openInterest'].fillna(0)
        total_df['put_OI'] = put_stats['openInterest'].fillna(0)
        total_df['call_GEX'] = call_stats['GEX'].fillna(0)
        total_df['put_GEX'] = put_stats['GEX'].fillna(0)
        total_df['total_GEX'] = total_df['call_GEX'] + total_df['put_GEX']
        
        # 7. Identify Walls
        # Call Wall: Strike with highest Net Call Open Interest
        call_wall = total_df['call_OI'].idxmax()
        
        # Put Wall: Strike with highest Net Put Open Interest
        put_wall = total_df['put_OI'].idxmax()
        
        # 8. Identify Gamma Flip
        # Find where cumulative GEX crosses zero
        cumulative_gex = total_df['total_GEX'].cumsum()
        signs = np.sign(cumulative_gex).diff().fillna(0)
        flips = signs[signs != 0].index
        
        if len(flips) > 0:
            # Find flip closest to current spot price
            gamma_flip = min(flips, key=lambda x: abs(x - spot))
        else:
            # Fallback: Strike with lowest absolute GEX
            gamma_flip = total_df['total_GEX'].abs().idxmin()

        logging.info(f"Done {ticker}: Spot={spot:.2f} | PutWall={put_wall} | CallWall={call_wall}")

        return {
            "spot": round(float(spot), 2),
            "callWall": int(call_wall),
            "putWall": int(put_wall),
            "gammaFlip": int(gamma_flip),
            "netGEX": round(total_df['total_GEX'].sum() / 10**9, 4), # Billions
            "updated": date.today().isoformat()
        }

    except Exception as e:
        logging.error(f"Error calculating walls for {ticker}: {e}")
        return None

# -----------------------------
# HELPER: Fetch ETF Holdings
# -----------------------------
def get_top_holdings(etf_symbol, n=10):
    """
    Fetches the top N holdings for a given ETF using yfinance.
    """
    yf_ticker = TICKER_MAP.get(etf_symbol, etf_symbol)
    try:
        ticker = yf.Ticker(yf_ticker)
        # yfinance creates a 'funds_data' object for ETFs
        holdings = ticker.funds_data.top_holdings
        
        if holdings is None or holdings.empty:
            logging.warning(f"No holdings data found for {etf_symbol}")
            return []
            
        # The index of the dataframe is the symbol
        top_symbols = holdings.index.tolist()[:n]
        return top_symbols
        
    except Exception as e:
        logging.error(f"Failed to fetch holdings for {etf_symbol}: {e}")
        return []

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    walls_dict = {}
    processed_tickers = set() # To avoid processing NVDA multiple times
    
    # 1. Process Main Indexes
    logging.info("--- Processing Indexes ---")
    for idx in INDEXES:
        if idx not in processed_tickers:
            res = get_gex_and_walls(idx)
            if res:
                walls_dict[idx] = res
            processed_tickers.add(idx)

    # 2. Process ETFs and their Holdings
    logging.info("--- Processing ETFs & Holdings ---")
    for etf in ETFS:
        # A. Process the ETF itself
        if etf not in processed_tickers:
            res = get_gex_and_walls(etf)
            if res:
                walls_dict[etf] = res
            processed_tickers.add(etf)
        
        # B. Get Holdings
        holdings = get_top_holdings(etf, n=TOP_HOLDINGS_COUNT)
        if holdings:
            logging.info(f"[{etf}] Holdings: {holdings}")
            
            # C. Process each holding
            for stock in holdings:
                if stock not in processed_tickers:
                    res_stock = get_gex_and_walls(stock)
                    if res_stock:
                        walls_dict[stock] = res_stock
                    processed_tickers.add(stock)
                    # Respectful sleep to avoid rate limiting
                    time.sleep(0.5) 
        else:
            logging.warning(f"Skipping holdings for {etf} (empty)")

    # 3. Export to JS
    js_text = "window.WALLS = " + json.dumps(walls_dict, indent=2) + ";"
    
    try:
        with open("options/walls.js", "w") as f:
            f.write(js_text)
        logging.info(f"SUCCESS: walls.js generated with {len(walls_dict)} tickers.")
    except FileNotFoundError:
        # Fallback if folder doesn't exist (for local testing)
        with open("walls.js", "w") as f:
            f.write(js_text)
        logging.info("SUCCESS: walls.js generated (root directory).")