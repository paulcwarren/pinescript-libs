import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

def run_return_stacking():
    initial_capital = 10000
    print("Downloading Return Stacking data (DBMF inception 2019)...")
    
    # Download both tickers
    tickers = ['QQQ', 'DBMF']
    data = yf.download(tickers, start='2019-05-08', progress=False)['Close']
    data.dropna(inplace=True)
    
    qqq_ret = data['QQQ'].pct_change().fillna(0)
    dbmf_ret = data['DBMF'].pct_change().fillna(0)
    
    # 1. Portfolio Construction
    # 100% QQQ + 100% DBMF = 200% Total Exposure
    # We must borrow 100% of our capital to fund the DBMF leg.
    
    # 2. Calculate Margin Borrowing Costs (4% Annualized)
    margin_rate = 0.04 / 252
    
    # 3. Calculate Strategy Returns
    # Return = QQQ Return + DBMF Return - Margin Interest
    strat_ret = qqq_ret + dbmf_ret - margin_rate
    
    # Cumulative values
    port_value = initial_capital * (1 + strat_ret).cumprod()
    qqq_value = initial_capital * (1 + qqq_ret).cumprod()
    
    # --- Performance Metrics ---
    total_days = len(strat_ret)
    years = total_days / 252
    
    total_return_strat = (port_value.iloc[-1] / initial_capital - 1) * 100
    total_return_qqq = (qqq_value.iloc[-1] / initial_capital - 1) * 100
    
    cagr = (port_value.iloc[-1] / initial_capital) ** (1 / years) - 1
    
    rolling_max = port_value.cummax()
    max_drawdown = ((port_value - rolling_max) / rolling_max).min()
    
    risk_free = 0.04 / 252
    excess_ret = strat_ret - risk_free
    sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    
    downside = excess_ret.copy()
    downside[downside > 0] = 0
    sortino = (excess_ret.mean() / downside.std()) * np.sqrt(252)

    print("\n=== STRATEGY HISTORICAL PERFORMANCE ===")
    print(f"Strategy:           Return Stacking (CTA Profile)")
    print(f"Allocation:         100% QQQ + 100% DBMF")
    print(f"Total Exposure:     200% (4% Margin Rate)")
    print(f"Starting Capital:   ${initial_capital:,.2f}")
    print(f"Ending Capital:     ${port_value.iloc[-1]:,.2f}")
    print(f"Total Return (RS):  {total_return_strat:.2f}%")
    print(f"Total Return (QQQ): {total_return_qqq:.2f}%")
    print(f"CAGR:               {cagr * 100:.2f}%")
    print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Sortino Ratio:      {sortino:.2f}")
    print("========================================\n")

    # =====================
    # Visualization
    # =====================
    print("Generating performance chart...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(port_value.index, port_value, label='Return Stacking (100% QQQ + 100% DBMF)', color='blue', linewidth=2)
    plt.plot(qqq_value.index, qqq_value, label='Buy & Hold QQQ', color='green', alpha=0.4)
    
    plt.title("Return Stacking (QQQ + DBMF) vs QQQ (Log Scale)")
    plt.yscale('log')
    plt.ylabel("Portfolio Value ($)")
    dollar_formatter = ticker.StrMethodFormatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(dollar_formatter)
    plt.gca().yaxis.set_minor_formatter(dollar_formatter)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_return_stacking()