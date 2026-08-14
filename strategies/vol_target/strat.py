import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings('ignore')

def run_volatility_targeting():
    initial_capital = 10000
    print("Downloading Volatility Targeting data...")
    
    # Start in 2019 to match DBMF's history for an apples-to-apples comparison
    data = yf.download('QQQ', start='2019-05-08', progress=False)['Close'].squeeze()
    data.dropna(inplace=True)
    
    daily_ret = data.pct_change().fillna(0)
    
    # 1. Calculate Current Market Volatility (20-day annualized realized volatility)
    # Require at least 20 days of data before calculating
    rolling_vol = daily_ret.rolling(window=20).std() * np.sqrt(252)
    
    # 2. Set Target Volatility
    target_vol = 0.25  # Targeting 25% annualized volatility
    
    # 3. Calculate Target Leverage
    # Target Volatility / Current Volatility. Cap at 2.0x leverage.
    leverage = (target_vol / rolling_vol).clip(upper=2.0)
    
    # Shift by 1 day to prevent look-ahead bias
    leverage = leverage.shift(1).fillna(1.0)
    
    # 4. Calculate Margin Borrowing Costs (Assuming 4% rate on borrowed funds)
    margin_rate = 0.04 / 252
    borrowed_amount = (leverage - 1.0).clip(lower=0.0)
    
    # 5. Calculate Strategy Returns
    strat_ret = (leverage * daily_ret) - (borrowed_amount * margin_rate)
    
    # Drop warmup period
    strat_ret = strat_ret.iloc[21:]
    daily_ret = daily_ret.iloc[21:]
    
    # Cumulative values
    port_value = initial_capital * (1 + strat_ret).cumprod()
    qqq_value = initial_capital * (1 + daily_ret).cumprod()
    
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
    print(f"Strategy:           Continuous Volatility Targeting")
    print(f"Underlying:         QQQ")
    print(f"Target Volatility:  25% Annualized")
    print(f"Max Leverage:       2.0x (4% Margin Rate)")
    print(f"Starting Capital:   ${initial_capital:,.2f}")
    print(f"Ending Capital:     ${port_value.iloc[-1]:,.2f}")
    print(f"Total Return (Vol): {total_return_strat:.2f}%")
    print(f"Total Return (QQQ): {total_return_qqq:.2f}%")
    print(f"CAGR:               {cagr * 100:.2f}%")
    print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Sortino Ratio:      {sortino:.2f}")
    print("========================================\n")

    # =====================
    # LIVE REBALANCE LINK GENERATOR
    # =====================
    # Fetch TQQQ price specifically for the retail ETF translation
    tqqq_data = yf.download('TQQQ', period='5d', progress=False)['Close']
    if isinstance(tqqq_data, pd.DataFrame): 
        tqqq_data = tqqq_data.squeeze()
    
    live_leverage = leverage.iloc[-1]
    qqq_price = data.iloc[-1]
    tqqq_price = tqqq_data.iloc[-1]

    # You can update this base URL to wherever you host the new HTML file
    base_url = "https://paulcwarren.github.io/pinescript-libs/strategies/vol_target/rebal.html" 
    final_link = f"{base_url}?lev={live_leverage:.2f}&qqq={qqq_price:.2f}&tqqq={tqqq_price:.2f}"

    print("=== COPY AND PASTE THIS INTO WHATSAPP ===")
    print("Yo fam! Wadup...It's time to rebalance your portfolios and keep making the wonga.  Here is your rebalance link:")
    print()
    print(final_link)
    print("=========================================\n")

    # =====================
    # Visualization
    # =====================
    print("Generating performance chart...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(port_value.index, port_value, label='Volatility Targeting (2.0x Max)', color='blue', linewidth=2)
    plt.plot(qqq_value.index, qqq_value, label='Buy & Hold QQQ', color='green', alpha=0.4)
    
    plt.title("Continuous Volatility Targeting vs QQQ (Log Scale)")
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
    run_volatility_targeting()