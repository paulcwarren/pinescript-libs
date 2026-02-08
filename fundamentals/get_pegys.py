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
        
        # 2. Yield (New Column)
        div_yield = (info.get('dividendYield') or 0) * 100
        
        # 3. Growth Rate (The culprit for N/A)
        # Try official growth first, then quarterly earnings growth, then 0
        growth = info.get('earningsGrowth')
        if growth is None:
            growth = info.get('earningsQuarterlyGrowth') or 0
        growth_pct = growth * 100
        
        # 4. Manual PEG Calculation (P/E divided by Growth)
        peg = round(pe / growth_pct, 2) if pe and growth_pct > 0 else None
        
        # 5. Manual PEGY Calculation (P/E divided by Growth + Yield)
        denominator = growth_pct + div_yield
        pegy = round(pe / denominator, 2) if pe and denominator > 0 else None
            
        return {
            "sector": info.get("sector", "N/A"),
            "etf": etf_ticker,
            "ticker": symbol,
            "name": info.get("shortName", "N/A"),
            "pe": round(pe, 2) if pe else "N/A",
            "growth": round(growth_pct, 2),
            "yield": round(div_yield, 2),
            "peg": peg if peg else "N/A",
            "pegy": pegy if pegy else "N/A"
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