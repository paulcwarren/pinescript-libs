import yfinance as yf
import pandas as pd
import sys

def get_felix_metrics(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # 1. Gross Margin (HG)
        gross_margin = info.get('grossMargins', 0) * 100
        
        # 2. Free Cash Flow (FCF)
        cashflow = stock.cashflow
        fcf = 0
        if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            fcf = cashflow.loc['Free Cash Flow'].iloc[0]
        
        # 3. ROIC (Return on Invested Capital)
        bs = stock.balance_sheet
        is_stmt = stock.financials
        roic = 0
        try:
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
    except Exception as e:
        # Silently fail for symbols like SESGL to avoid clutter
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <ETF_TICKER>")
        return

    etf_ticker = sys.argv[1].upper()
    print(f"--- Analyzing Sector ETF: {etf_ticker} ---")
    
    etf = yf.Ticker(etf_ticker)
    try:
        holdings_df = etf.funds_data.top_holdings
        top_10 = holdings_df.index.tolist()[:10]
    except:
        print("Error: Could not retrieve holdings.")
        return

    results = []
    for t in top_10:
        print(f"Fetching metrics for {t}...")
        data = get_felix_metrics(t)
        if data:
            results.append(data)

    # --- THE FIX: SIMPLE NUMERICAL SORT BY FCF ---
    df = pd.DataFrame(results)
    
    # Sort descending by FCF (M)
    df = df.sort_values(by='FCF (M)', ascending=False)

    print("\n--- Top Performers (Ranked by Free Cash Flow) ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()