import os
import json
import time
import logging
import yfinance as yf
import numpy as np

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configuration Paths
JSON_INPUT_PATH = "etfs.json"
JSON_OUTPUT_PATH = "options/walls.js"
TOP_HOLDINGS_COUNT = 10

def load_tickers():
    """Loads the ticker configuration structure from etfs.json."""
    if not os.path.exists(JSON_INPUT_PATH):
        logging.error(f"Input configuration file missing at {JSON_INPUT_PATH}")
        return {}
    with open(JSON_INPUT_PATH, 'r') as f:
        return json.load(f)

def get_top_holdings(etf_ticker, count):
    """
    Placeholder/Stub for fetching top ETF holdings. 
    Replace this logic with your actual holdings scraper or data provider if needed.
    """
    # Quick static map example for safety if dynamic scraper fails
    mock_holdings = {
        "UFO": ["PL", "MDA.TO", "GRMN", "VSAT", "SATS", "RKLB", "SESGL", "SIRI", "ASTS", "9412.T"],
        "XLK": ["MSFT", "AAPL", "AVGO", "NVDA", "AMD", "CSCO", "CRM", "ORCL", "PANW", "INTU"]
    }
    return mock_holdings.get(etf_ticker, [])[:count]

def get_gex_and_walls(ticker_symbol):
    """
    Scans the options chain via yfinance.
    Returns near-term decay-weighted walls and absolute macro anchor walls.
    """
    ticker_obj = yf.Ticker(ticker_symbol)
    
    try:
        # Get historical/current spot price to orient calculations
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            logging.warning(f"No price data found for {ticker_symbol}")
            return None
        spot_price = hist['Close'].iloc[-1]
    except Exception as e:
        logging.error(f"Failed to fetch price data for {ticker_symbol}: {e}")
        return None

    try:
        all_expirations = ticker_obj.options
        if not all_expirations:
            return None
    except Exception as e:
        logging.error(f"Failed to fetch options chain structure for {ticker_symbol}: {e}")
        return None

    unweighted_call_gamma = {}
    unweighted_put_gamma = {}
    weighted_call_gamma = {}
    weighted_put_gamma = {}
    
    net_gex = 0.0

    # Look through expiration dates (Up to 3 iterations for weighted near-term profiling)
    for idx, exp in enumerate(all_expirations[:5]):  # Scan up to 5 expiries for absolute anchors
        # Establish your precise 3-day time-decay multipliers
        weight = 1.0 if idx == 0 else (0.5 if idx == 1 else (0.25 if idx == 2 else 0.0))
        
        try:
            opt = ticker_obj.option_chain(exp)
        except Exception:
            # Skip faulty specific option chains silently to move fast
            continue

        # --- Process Calls ---
        if not opt.calls.empty:
            for _, row in opt.calls.iterrows():
                strike = float(row['strike'])
                # Use Open Interest as raw proxy metric if explicit Gamma calculations are absent
                gamma = float(row.get('gamma', row['openInterest']))
                if np.isnan(gamma): 
                    gamma = float(row['openInterest'])

                # Track Net GEX (Calls are positive gamma exposure for dealers)
                net_gex += gamma
                
                # 1. Anchor Track (Absolute unweighted accumulation across the chain)
                unweighted_call_gamma[strike] = unweighted_call_gamma.get(strike, 0.0) + gamma
                
                # 2. Tactical Track (Near-term decaying flow)
                if weight > 0:
                    weighted_call_gamma[strike] = weighted_call_gamma.get(strike, 0.0) + (gamma * weight)

        # --- Process Puts ---
        if not opt.puts.empty:
            for _, row in opt.puts.iterrows():
                strike = float(row['strike'])
                gamma = float(row.get('gamma', row['openInterest']))
                if np.isnan(gamma): 
                    gamma = float(row['openInterest'])

                # Track Net GEX (Puts represent negative gamma exposure positions)
                net_gex -= gamma
                
                # 1. Anchor Track (Absolute unweighted accumulation across the chain)
                unweighted_put_gamma[strike] = unweighted_put_gamma.get(strike, 0.0) + gamma
                
                # 2. Tactical Track (Near-term decaying flow)
                if weight > 0:
                    weighted_put_gamma[strike] = weighted_put_gamma.get(strike, 0.0) + (gamma * weight)

    # Determine execution levels based on peak volume/gamma clusters
    tactical_call = max(weighted_call_gamma, key=weighted_call_gamma.get) if weighted_call_gamma else None
    tactical_put = max(weighted_put_gamma, key=weighted_put_gamma.get) if weighted_put_gamma else None
    
    anchor_call = max(unweighted_call_gamma, key=unweighted_call_gamma.get) if unweighted_call_gamma else None
    anchor_put = max(unweighted_put_gamma, key=unweighted_put_gamma.get) if unweighted_put_gamma else None

    # Classify overall market dealer regime positioning format
    if net_gex > 0.5:
        outlook = "STABLE / GRIND (Long Gamma)"
    elif net_gex < -0.5:
        outlook = "VOLATILE / DANGER (Short Gamma)"
    else:
        outlook = "VOLATILE / TRANSITION"

    return {
        "spot": round(spot_price, 2),
        "net_gex_bn": round(net_gex / 1_000_000, 4), # Normalized to billions string output scale
        "outlook": outlook,
        "tactical": {"call": tactical_call, "put": tactical_put},
        "anchor": {"call": anchor_call, "put": anchor_put}
    }

def main():
    config = load_tickers()
    if not config:
        return

    walls_dict = {}
    processed_tickers = set()

    # Flatten sections cleanly for sequencing loops
    categories = [
        ("Indexes", config.get("INDEXES", [])),
        ("ETFs", config.get("ETFS", [])),
        ("Extra Watchlist", config.get("EXTRA_TICKERS", []))
    ]

    for label, ticker_list in categories:
        logging.info(f"--- Processing {label} ---")
        for ticker in ticker_list:
            if not ticker or ticker in processed_tickers:
                continue
            
            # 1. Process Core Asset Base Level
            try:
                res = get_gex_and_walls(ticker)
                if res:
                    walls_dict[ticker] = res
                    logging.info(
                        f"{ticker:5s} | Spot: {res['spot']:<7} | GEX: {res['net_gex_bn']:<7}bn | "
                        f"Tactical C/P: {str(res['tactical']['call'])+'/'+str(res['tactical']['put']):<13} | "
                        f"Anchor C/P: {str(res['anchor']['call'])+'/'+str(res['anchor']['put'])}"
                    )
                else:
                    logging.warning(f"Skipping {ticker}: No alternative parameters or derivatives found.")
            except Exception as e:
                logging.error(f"Skipping main ticker symbol target {ticker} due to active error run context: {e}")
            
            processed_tickers.add(ticker)
            time.sleep(0.5)

            # 2. Process Deep Sector Holdings Branches (For designated ETF tracking)
            if label == "ETFs":
                holdings = get_top_holdings(ticker, TOP_HOLDINGS_COUNT)
                for stock in holdings:
                    if not stock or stock in processed_tickers:
                        continue
                    
                    try:
                        h_res = get_gex_and_walls(stock)
                        if h_res:
                            walls_dict[stock] = h_res
                            logging.info(
                                f"  -> {stock:5s} (Holding) | Tactical C/P: {str(h_res['tactical']['call'])+'/'+str(h_res['tactical']['put']):<13}"
                            )
                    except Exception as e:
                        logging.error(f"Skipping holding asset node {stock} in {ticker} context framework loop error: {e}")
                    
                    processed_tickers.add(stock)
                    time.sleep(0.5)

    # Export cleanly formatted object literal payload targeting web visualization UI arrays
    try:
        with open(JSON_OUTPUT_PATH, 'w') as out_file:
            out_file.write(f"const wallsData = {json.dumps(walls_dict, indent=2)};")
        logging.info(f"SUCCESS: {JSON_OUTPUT_PATH} generated with {len(walls_dict)} processed tickers.")
    except Exception as e:
        logging.error(f"Failed writing payload structures back down to disk: {e}")

if __name__ == "__main__":
    main()