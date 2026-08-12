import pandas as pd
import argparse
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

def get_market_cap(ticker):
    """Fetches market cap safely handling None types and rate limits."""
    try:
        mc = yf.Ticker(ticker).fast_info['marketCap']
        if mc is None:
            return ticker, 0
        return ticker, float(mc)
    except Exception:
        return ticker, 0

def main():
    parser = argparse.ArgumentParser(description="Download and filter US stock tickers.")
    parser.add_argument('--top', type=int, help="Limit to the N most important stocks by Market Cap (e.g., 1000)")
    # 'extend' allows repeating the flag: --format csv --format line
    parser.add_argument('--format', choices=['csv', 'line'], nargs='+', action='extend',
                        help="Output format(s): 'csv' (quoted, comma-separated) and/or 'line' (unquoted, one per line).")
    args = parser.parse_args()

    # Standardize formats to avoid duplicates while retaining default behavior
    formats = args.format if args.format else ['csv']
    formats = list(dict.fromkeys(formats))

    print("Fetching raw data from NASDAQ FTP...")
    
    # 1. Load the raw pipe-delimited data
    nasdaq = pd.read_csv("ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt", sep="|")
    other = pd.read_csv("ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt", sep="|")

    # 2. Align the primary ticker column names
    nasdaq = nasdaq.rename(columns={"Symbol": "Ticker"})
    other = other.rename(columns={"ACT Symbol": "Ticker"})

    # 3. Append the datasets
    df = pd.concat([nasdaq, other], ignore_index=True)

    # 4. Clean system rows and test issues
    df = df.dropna(subset=['Ticker']) 
    df = df[df['Test Issue'] == 'N']

    # 5. Filter out non-alphabetical tickers
    df = df[~df['Ticker'].str.contains(r'[^A-Za-z]', regex=True, na=False)]

    # 6. Filter Security Names for junk assets
    junk_keywords = 'Warrant|Wts|Right|Unit|Preferred|%'
    df = df[~df['Security Name'].str.contains(junk_keywords, case=False, na=False, regex=True)]

    # 7. Drop ETFs
    df = df[df['ETF'] == 'N']

    # 8. Extract the tickers to a Python list
    clean_tickers = df['Ticker'].tolist()

    # Calculate Market Caps if requested
    if args.top:
        print(f"Fetching market caps for {len(clean_tickers)} tickers to determine the top {args.top}...")
        print("Using throttled concurrent threads to avoid Yahoo rate limits. Please wait...")
        
        market_caps = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(get_market_cap, clean_tickers)
            for result in results:
                market_caps.append(result)
        
        # Sort the list of tuples by market cap descending
        market_caps.sort(key=lambda x: x[1], reverse=True)
        
        # Slice the top N and extract just the ticker strings
        clean_tickers = [x[0] for x in market_caps[:args.top]]

    # 9. Loop through the requested formats and generate outputs
    for fmt in formats:
        if fmt == 'csv':
            output_data = ",".join(f'"{ticker}"' for ticker in clean_tickers)
            ext = ".csv"
        else:
            output_data = "\n".join(clean_tickers) + "\n"
            ext = ".txt"

        filename = f"top{args.top}_listed{ext}" if args.top else f"alllisted{ext}"
        
        with open(filename, "w") as f:
            f.write(output_data)

        print(f"Success! Exported {len(clean_tickers)} individual stock tickers to {filename} ({fmt} format)")

if __name__ == "__main__":
    main()