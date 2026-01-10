import json
import os
import yfinance as yf

# 1. Define the dictionary first!
mapping = {}

sector_etfs = ["BLOK", "IGV", "CLOU", "MAGS", "QTUM", "URA", "UFO", "ROBO", "OIH", "XLK", "XLF", "XLY", "XLI", "XLE", "XLC", "XLV", "XLU", "XLRE", "XHB", "XBI", "XLP"]

print("--- Starting Scrape ---")
for etf in sector_etfs:
    try:
        print(f"Processing {etf}...")
        t = yf.Ticker(etf)
        holdings = t.funds_data.top_holdings
        
        if holdings is not None and not holdings.empty:
            # Clean tickers: remove non-alphanumeric chars to keep Pine Script happy
            raw_tickers = holdings.index.tolist()[:10]
            clean_tickers = [str(tk).split()[0].strip() for tk in raw_tickers] 
            mapping[etf] = clean_tickers
            print(f"✅ {etf}: Found {len(clean_tickers)} tickers")
        else:
            if etf == "MAGS":
                mapping[etf] = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
                print(f"ℹ️ {etf}: Used manual fallback")
            else:
                mapping[etf] = []
                print(f"⚠️ {etf}: No holdings found")
                
    except Exception as e:
        print(f"❌ {etf}: Error - {e}")
        mapping[etf] = []

# 2. Save the results
os.makedirs('data', exist_ok=True)
with open('data/holdings.json', 'w') as f:
    json.dump(mapping, f, indent=4)

print(f"--- Scrape Complete: {len(mapping)} sectors processed ---")