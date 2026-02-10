import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
import json
from datetime import date, datetime
import time
import os

# -----------------------------
# CONFIGURATION & DATA LOADING
# -----------------------------
def load_config():
    """Loads ticker lists from etfs.json with fail-safes."""
    try:
        with open('etfs.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("etfs.json not found! Using empty defaults.")
        return {"INDEXES": [], "ETFS": [], "EXTRA_TICKERS": []}

# Global constants for the math/logic
TOP_HOLDINGS_COUNT = 10   # Fetch top 10 stocks for each ETF
EXPIRATION_LOOKAHEAD = 3  # Aggregate OI across next 3 expirations

TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# MATH: Black-Scholes Gamma
# -----------------------------
def calc_gamma(S, K, sigma, T, r=0.015):
    T = np.maximum(T, 0.001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma_val

# -----------------------------
# CORE LOGIC: Process Option Chain
# -----------------------------
def get_gex_and_walls(ticker):
    yf_ticker = TICKER_MAP.get(ticker, ticker)
    stock = yf.Ticker(yf_ticker)
    
    try:
        hist = stock.history(period="1d")
        if hist.empty:
            logging.warning(f"No price data for {ticker}")
            return None
        spot = hist['Close'].iloc[-1]
        
        exps = stock.options
        if not exps:
            return None
            
        target_exps = exps[:EXPIRATION_LOOKAHEAD]
        all_calls, all_puts = [], []

        for exp_date_str in target_exps:
            try:
                chain = stock.option_chain(exp_date_str)
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                days_to_expiry = (exp_date - date.today()).days
                T = max(days_to_expiry, 0.5) / 365.0
                
                c, p = chain.calls.copy(), chain.puts.copy()
                c['type'], c['T'] = 'call', T
                p['type'], p['T'] = 'put', T
                all_calls.append(c); all_puts.append(p)
            except:
                continue

        if not all_calls or not all_puts:
            return None

        df_calls, df_puts = pd.concat(all_calls), pd.concat(all_puts)
        df_calls['impliedVolatility'] = df_calls['impliedVolatility'].replace(0, np.nan).fillna(0.2)
        df_puts['impliedVolatility'] = df_puts['impliedVolatility'].replace(0, np.nan).fillna(0.2)
        
        df_calls['gamma'] = calc_gamma(spot, df_calls['strike'], df_calls['impliedVolatility'], df_calls['T'])
        df_calls['GEX'] = df_calls['gamma'] * df_calls['openInterest'] * (spot**2) * -1 
        
        df_puts['gamma'] = calc_gamma(spot, df_puts['strike'], df_puts['impliedVolatility'], df_puts['T'])
        df_puts['GEX'] = df_puts['gamma'] * df_puts['openInterest'] * (spot**2) 
        
        call_stats = df_calls.groupby('strike')[['openInterest', 'GEX']].sum()
        put_stats = df_puts.groupby('strike')[['openInterest', 'GEX']].sum()
        
        total_df = pd.DataFrame(index=sorted(set(call_stats.index) | set(put_stats.index)))
        total_df['call_OI'] = call_stats['openInterest'].fillna(0)
        total_df['put_OI'] = put_stats['openInterest'].fillna(0)
        total_df['total_GEX'] = (call_stats['GEX'].fillna(0) + put_stats['GEX'].fillna(0))
        
        # Day Trading Bounds (+/- 10%)
        l_bound, u_bound = spot * 0.90, spot * 1.10
        
        c_cands = total_df[(total_df.index >= spot) & (total_df.index <= u_bound)]
        call_wall = c_cands['call_OI'].idxmax() if not c_cands.empty else total_df['call_OI'].idxmax()

        p_cands = total_df[(total_df.index >= l_bound) & (total_df.index <= spot)]
        put_wall = p_cands['put_OI'].idxmax() if not p_cands.empty else total_df['put_OI'].idxmax()

        cumulative_gex = total_df['total_GEX'].cumsum()
        signs = np.sign(cumulative_gex).diff().fillna(0)
        flips = signs[signs != 0].index
        gamma_flip = min(flips, key=lambda x: abs(x - spot)) if len(flips) > 0 else total_df['total_GEX'].abs().idxmin()

        logging.info(f"{ticker}: Spot {spot:.2f} | PutWall {put_wall} | CallWall {call_wall}")

        return {
            "spot": round(float(spot), 2),
            "callWall": int(call_wall),
            "putWall": int(put_wall),
            "gammaFlip": int(gamma_flip),
            "netGEX": round(total_df['total_GEX'].sum() / 10**9, 4),
            "updated": date.today().isoformat()
        }
    except Exception as e:
        logging.error(f"Error for {ticker}: {e}")
        return None

# -----------------------------
# HELPER: Fetch Holdings
# -----------------------------
def get_top_holdings(etf_symbol, n=10):
    yf_ticker = TICKER_MAP.get(etf_symbol, etf_symbol)
    try:
        t = yf.Ticker(yf_ticker)
        holdings = t.funds_data.top_holdings
        return [] if holdings is None or holdings.empty else holdings.index.tolist()[:n]
    except:
        return []

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Load configuration from JSON
    config = load_config()
    index_list = config.get("INDEXES", [])
    etf_list = config.get("ETFS", [])
    extra_tickers = config.get("EXTRA_TICKERS", [])

    walls_dict = {}
    processed_tickers = set()

    # Define a helper to run the processing logic to avoid code repetition
    def process_list(ticker_list, label, handle_holdings=False):
        logging.info(f"--- Processing {label} ---")
        for ticker in ticker_list:
            if ticker not in processed_tickers:
                res = get_gex_and_walls(ticker)
                if res:
                    walls_dict[ticker] = res
                processed_tickers.add(ticker)
                
                if handle_holdings:
                    holdings = get_top_holdings(ticker, TOP_HOLDINGS_COUNT)
                    if holdings:
                        logging.info(f"[{ticker}] Holdings: {holdings}")
                        for stock in holdings:
                            if stock not in processed_tickers:
                                h_res = get_gex_and_walls(stock)
                                if h_res:
                                    walls_dict[stock] = h_res
                                processed_tickers.add(stock)
                                time.sleep(0.5)
                time.sleep(0.5)

    # Run the sequences
    process_list(index_list, "Indexes")
    process_list(etf_list, "ETFs", handle_holdings=True)
    process_list(extra_tickers, "Extra Watchlist")

    # Export to JS
    js_text = "window.WALLS = " + json.dumps(walls_dict, indent=2) + ";"
    output_path = "options/walls.js" if os.path.exists("options") else "walls.js"
    
    with open(output_path, "w") as f:
        f.write(js_text)
    logging.info(f"SUCCESS: {output_path} generated with {len(walls_dict)} tickers.")