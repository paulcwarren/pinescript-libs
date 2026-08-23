import json
import os
import re
import yfinance as yf

# 1. Define the dictionary first
mapping = {}

# Indices first, then major GICS sectors, then thematic ETFs
sector_etfs = [
    "SPY", "QQQ", "IWM", "DIA", 
    "XLK", "XLF", "XLY", "XLI", "XLE", "XLC", "XLV", "XLU", "XLRE", "XLP",
    "BLOK", "IGV", "CLOU", "MAGS", "QTUM", "URA", "UFO", "ROBO", "OIH", "SOXX", "XME", "XRT", "XHB", "XBI", "XLB"
]

# Keep track of tickers we have already assigned to a sector
seen_tickers = set()

print("--- Starting Scrape ---")
for etf in sector_etfs:
    try:
        print(f"Processing {etf}...")
        
        # Handle manual indices or ETFs if they aren't standard yfinance lookups
        if etf in ["SPY", "QQQ", "IWM", "DIA"]:
            if etf == "SPY": raw_tickers = ["SPY"]
            elif etf == "QQQ": raw_tickers = ["QQQ"]
            elif etf == "IWM": raw_tickers = ["IWM"]
            elif etf == "DIA": raw_tickers = ["DIA"]
        else:
            t = yf.Ticker(etf)
            holdings = t.funds_data.top_holdings
            raw_tickers = holdings.index.tolist()[:10] if (holdings is not None and not holdings.empty) else []

        clean_tickers = []
        for tk in raw_tickers:
            # Remove spaces/swap info
            s = str(tk).split()[0] 
            # Clean ticker to match Pine cleanT (removes .T, -B, etc.)
            s = re.sub(r'[.\-].*$', '', s) 
            
            # Only add the ticker if it hasn't appeared in a prior sector yet
            if s not in seen_tickers:
                seen_tickers.add(s)
                clean_tickers.append(s)
        
        # Fallback for MAGS if needed and empty
        if not clean_tickers and etf == "MAGS":
            mags_fallback = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
            for m in mags_fallback:
                if m not in seen_tickers:
                    seen_tickers.add(m)
                    clean_tickers.append(m)
            print(f"ℹ️ {etf}: Used manual fallback")

        mapping[etf] = clean_tickers
        print(f"✅ {etf}: Found {len(clean_tickers)} unique tickers")
                
    except Exception as e:
        print(f"❌ {etf}: Error - {e}")
        mapping[etf] = []

# 2. Save the results
os.makedirs('data', exist_ok=True)
with open('data/holdings.json', 'w') as f:
    json.dump(mapping, f, indent=4)

print(f"--- Scrape Complete: {len(mapping)} sectors processed ---")
