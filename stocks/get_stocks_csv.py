import pandas as pd
import argparse
import requests


def parse_market_cap(value):
    """Converts string multipliers (e.g., '10B', '500M') to numeric float values."""
    if not value:
        return 0

    value = str(value).upper().strip()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}

    if value[-1] in multipliers:
        try:
            return float(value[:-1]) * multipliers[value[-1]]
        except ValueError:
            return 0

    try:
        return float(value)
    except ValueError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download and filter US stock tickers.")
    parser.add_argument('--top', type=int, help="Limit to the N most important stocks by Market Cap (e.g., 1000)")
    parser.add_argument('--min-market-cap', type=str, help="Minimum market cap (e.g., 10B, 500M, 2T)")
    parser.add_argument('--format', choices=['csv', 'line'], nargs='+', action='extend',
                        help="Output format(s): 'csv' and/or 'line'.")
    args = parser.parse_args()

    formats = args.format if args.format else ['csv']
    formats = list(dict.fromkeys(formats))

    print("Fetching full screener data directly from the NASDAQ API...")

    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to fetch data from NASDAQ: {e}")
        return

    rows = data.get('data', {}).get('rows', [])

    if not rows:
        print("No data returned from NASDAQ API.")
        return

    df = pd.DataFrame(rows)

    # Standardize column names
    df = df.rename(columns={
        "symbol": "Ticker",
        "name": "Security Name"
    })

    df['Ticker'] = df['Ticker'].str.strip()

    # Filter out non-alphabetical tickers
    df = df[~df['Ticker'].str.contains(r'[^A-Za-z]', regex=True, na=False)]

    # Filter junk assets
    junk_keywords = 'Warrant|Wts|Right|Unit|Preferred|%'
    df = df[~df['Security Name'].str.contains(
        junk_keywords, case=False, na=False, regex=True
    )]

    # Remove ETFs and Funds
    etf_keywords = 'ETF|Fund|Trust|Portfolio|Invesco|iShares|ProShares|Vanguard|SPDR|Direxion'
    df = df[~df['Security Name'].str.contains(
        etf_keywords, case=False, na=False, regex=True
    )]

    # Clean market cap
    df['marketCap'] = df['marketCap'].replace({',': ''}, regex=True)
    df['marketCap'] = pd.to_numeric(
        df['marketCap'], errors='coerce'
    ).fillna(0)

    # Apply minimum market cap
    if args.min_market_cap:
        min_mc = parse_market_cap(args.min_market_cap)

        print(f"Filtering for stocks with Market Cap >= {min_mc:,.0f}...")

        df = df[df['marketCap'] >= min_mc]

    # Sort largest first
    df = df.sort_values(by='marketCap', ascending=False)

    # Apply top-N
    if args.top:
        print(f"Selecting the top {args.top} stocks by Market Cap...")
        df = df.head(args.top)

    clean_tickers = df['Ticker'].tolist()

    if not clean_tickers:
        print("No tickers matched your criteria.")
        return

    # Write requested formats
    for fmt in formats:

        if fmt == 'csv':
            output_df = df[['Ticker', 'Security Name', 'marketCap']].copy()
            output_df = output_df.rename(columns={'marketCap': 'Market Cap'})

            ext = ".csv"
            output_df.to_csv(
                f"min{args.min_market_cap}_listed{ext}" if args.min_market_cap
                else f"top{args.top}_listed{ext}" if args.top
                else f"alllisted{ext}",
                index=False
            )

        else:
            output_data = "\n".join(clean_tickers) + "\n"
            ext = ".txt"

            if args.top:
                filename = f"top{args.top}_listed{ext}"
            elif args.min_market_cap:
                filename = f"min{args.min_market_cap}_listed{ext}"
            else:
                filename = f"alllisted{ext}"

            with open(filename, "w") as f:
                f.write(output_data)

        filename = (
            f"min{args.min_market_cap}_listed{ext}"
            if args.min_market_cap
            else f"top{args.top}_listed{ext}"
            if args.top
            else f"alllisted{ext}"
        )

        print(
            f"Success! Exported {len(clean_tickers)} stocks to "
            f"{filename} ({fmt} format)"
        )


if __name__ == "__main__":
    main()