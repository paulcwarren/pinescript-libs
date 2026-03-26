import yfinance as yf
import pandas as pd
import argparse
import sys

def get_felix_metrics(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # 1. Gross Margin
        gross_margin = info.get('grossMargins', 0) * 100
        
        # 2. Free Cash Flow (FCF)
        cashflow = stock.cashflow
        fcf = 0
        if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            fcf = cashflow.loc['Free Cash Flow'].iloc[0]
        
        # 3. ROIC (Return on Invested Capital)
        roic = 0
        try:
            bs = stock.balance_sheet
            is_stmt = stock.financials
            net_income = is_stmt.loc['Net Income'].iloc[0]
            total_assets = bs.loc['Total Assets'].iloc[0]
            curr_liab = bs.loc['Current Liabilities'].iloc[0]
            invested_capital = total_assets - curr_liab
            roic = (net_income / invested_capital) * 100
        except:
            roic = info.get('returnOnEquity', 0) * 100 # Fallback to ROE

        return {
            "Ticker": ticker_symbol,
            "FCF (M)": round(fcf / 1e6, 2),
            "Gross Margin %": round(gross_margin, 2),
            "ROIC %": round(roic, 2)
        }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze stock fundamentals.")
    # Add optional -i flags that can be repeated
    parser.add_argument('-i', '--individual', action='append', help="Individual tickers to analyze")
    # Add positional argument for ETF (optional if -i is used)
    parser.add_argument('etf', nargs='?', help="ETF ticker to pull top 10 holdings from")

    args = parser.parse_args()

    tickers_to_process = []

    # Case 1: Individual tickers provided
    if args.individual:
        tickers_to_process = [t.upper() for t in args.individual]
        print(f"--- Analyzing Individual Tickers: {', '.join(tickers_to_process)} ---")
    
    # Case 2: ETF provided
    elif args.etf:
        etf_ticker = args.etf.upper()
        print(f"--- Analyzing Sector ETF: {etf_ticker} ---")
        etf = yf.Ticker(etf_ticker)
        try:
            # Note: funds_data depends on current yfinance version support for the specific ETF
            holdings_df = etf.funds_data.top_holdings
            tickers_to_process = holdings_df.index.tolist()[:10]
        except Exception:
            print("Error: Could not retrieve holdings for this ETF.")
            return
    else:
        parser.print_help()
        return

    results = []
    for t in tickers_to_process:
        print(f"Fetching metrics for {t}...")
        data = get_felix_metrics(t)
        if data:
            results.append(data)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='FCF (M)', ascending=False)
        print("\n--- Fundamentals Ranking (By Free Cash Flow) ---")
        print(df.to_string(index=False))
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    main()