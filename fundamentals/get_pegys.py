import yfinance as yf
import pandas as pd
import json
import time

def load_config():
    with open('etfs.json', 'r') as f:
        return json.load(f)

def get_top_holdings(etf_symbol, n=15):
    try:
        t = yf.Ticker(etf_symbol)
        holdings = t.funds_data.top_holdings
        if holdings is None or holdings.empty:
            return []
        return holdings.index.tolist()[:n]
    except:
        return []

def get_pegy_data(symbol, etf_ticker):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 1. P/E Ratio
        pe = info.get('forwardPE') or info.get('trailingPE')
        
        # 2. Yield Logic - Normalize to whole number (e.g., 3.5 instead of 0.035)
        raw_yield = info.get('dividendYield') or 0
        if raw_yield < 1.0 and raw_yield != 0:
            div_yield = raw_yield * 100
        else:
            div_yield = raw_yield
        
        # 3. Growth Logic - Normalize to whole number (e.g., 15.0 instead of 0.15)
        raw_growth = info.get('earningsGrowth')
        if raw_growth is None:
            raw_growth = info.get('earningsQuarterlyGrowth') or 0
        
        if abs(raw_growth) < 1.0 and raw_growth != 0:
            growth_pct = raw_growth * 100
        else:
            growth_pct = raw_growth
        
        # 4. Denominator for PEGY
        denominator = growth_pct + div_yield
        
        # 5. PEG & PEGY Logic
        peg = round(pe / growth_pct, 2) if pe and growth_pct > 0 else "Contraction"
        pegy = round(pe / denominator, 2) if pe and denominator > 0 else "Contraction"
            
        return {
            "sector": info.get("sector", "N/A"),
            "etf": etf_ticker,
            "ticker": symbol,
            "name": info.get("shortName", "N/A"),
            "pe": round(pe, 2) if pe else "N/A",
            "growth": round(growth_pct, 2),
            "yield": round(div_yield, 2),
            "peg": peg,
            "pegy": pegy
        }
    except:
        return None
        
def main():
    config = load_config()
    etf_list = config.get("ETFS", [])
    extra_tickers = config.get("EXTRA_TICKERS", [])
    
    ticker_to_etf = {ticker: "Watchlist" for ticker in extra_tickers}
    
    for etf in etf_list:
        print(f"Scanning ETF: {etf}")
        holdings = get_top_holdings(etf, n=15)
        for h in holdings:
            if h not in ticker_to_etf:
                ticker_to_etf[h] = etf
        time.sleep(0.1)

    results = []
    for symbol, etf_ticker in ticker_to_etf.items():
        if not symbol: continue
        print(f"Data for {symbol}...")
        data = get_pegy_data(symbol, etf_ticker)
        if data:
            results.append(data)
        time.sleep(0.3)

    df = pd.DataFrame(results)
    if not df.empty:
        # Columns in requested order
        cols = ["sector", "etf", "ticker", "name", "pe", "growth", "yield", "peg", "pegy"]
        df = df[cols]
        df.sort_values(by="pegy", key=lambda x: pd.to_numeric(x, errors='coerce'), inplace=True)
        df.to_csv("pegy_results.csv", index=False)
        print("Success: Report generated with yields and manual PEGs.")

if __name__ == "__main__":
    main()