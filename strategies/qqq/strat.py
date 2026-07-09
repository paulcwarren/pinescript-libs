import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

# Suppress yfinance multi-index/formatting warnings for clean output
warnings.filterwarnings('ignore')

# =====================
# 1. Configuration & Data Fetching
# =====================
start_date = "2020-12-02" 
primary_ticker = "QQQ"
hedge_ticker = "DBMF"
initial_capital = 10000

# --- STRATEGY FLAGS ---
allocation_profile = "Moderate"     # Options: "Conservative", "Moderate", "Aggressive", "Insane"
rebalance_freq = "Weekly"           # Options: "Daily", "Weekly"
ma_type = "SMA"                     # Options: "SMA", "EMA"

# Volatility Circuit Breaker Settings
enable_vol_override = False         # Preemptively cut leverage during high volatility
vol_override_cap = 0.75             # Max QQQ allocation during extreme vol
# --------------------------

print(f"Downloading data for {primary_ticker} and {hedge_ticker}...")
data = yf.download([primary_ticker, hedge_ticker], start=start_date, progress=False)

# Separate into usable dataframes
if isinstance(data.columns, pd.MultiIndex):
    qqq = data.xs(primary_ticker, level=1, axis=1).copy()
    hedge = data.xs(hedge_ticker, level=1, axis=1).copy()
else:
    qqq = data[primary_ticker].copy() # Fallback for single tickers if structured differently
    hedge = data[hedge_ticker].copy()

# =====================
# 2. Replicate Pine Script Indicators
# =====================
ma_len = 100
atr_len = 14
slope_lookback = 5
vol_len = 50

# Core Moving Average Math
if ma_type.upper() == "EMA":
    qqq['MA'] = qqq['Close'].ewm(span=ma_len, adjust=False).mean()
else:
    qqq['MA'] = qqq['Close'].rolling(window=ma_len).mean()

qqq['MA_Slope'] = (qqq['MA'] - qqq['MA'].shift(slope_lookback)) / qqq['MA'].shift(slope_lookback)

# ATR Calculation
tr1 = qqq['High'] - qqq['Low']
tr2 = (qqq['High'] - qqq['Close'].shift(1)).abs()
tr3 = (qqq['Low'] - qqq['Close'].shift(1)).abs()
qqq['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
qqq['ATR'] = qqq['TR'].ewm(alpha=1/atr_len, adjust=False).mean()

# Calculate Volatility Mean and Standard Deviation
qqq['Vol'] = qqq['ATR'] / qqq['Close']
qqq['Vol_Mean'] = qqq['Vol'].rolling(window=vol_len).mean()
qqq['Vol_Std'] = qqq['Vol'].rolling(window=vol_len).std()

# Regime 0 uses standard expansion (bear market filter)
qqq['Vol_Expand'] = qqq['Vol'] > qqq['Vol_Mean'] 

# Circuit Breaker requires EXTREME expansion (2 standard deviations)
qqq['Vol_Extreme'] = qqq['Vol'] > (qqq['Vol_Mean'] + (qqq['Vol_Std'] * 2))

# =====================
# 3. Regime Logic & Target Allocations
# =====================
cond_close_above = qqq['Close'] > qqq['MA']
cond_slope_up = qqq['MA_Slope'] > 0

qqq['Regime'] = 1 
qqq.loc[~cond_close_above & cond_slope_up, 'Regime'] = 2  
qqq.loc[cond_close_above & ~cond_slope_up, 'Regime'] = 3  
qqq.loc[cond_close_above & cond_slope_up, 'Regime'] = 4   
qqq.loc[~cond_close_above & ~cond_slope_up & qqq['Vol_Expand'], 'Regime'] = 0  

# Profile Mapping
if allocation_profile == "Conservative":
    alloc_map = { 4: 1.20, 3: 1.05, 2: 0.90, 1: 0.50, 0: 0.00 }
elif allocation_profile == "Aggressive":
    alloc_map = { 4: 1.50, 3: 1.25, 2: 1.00, 1: 0.75, 0: 0.50 }
elif allocation_profile == "Insane":
    alloc_map = { 4: 2.0, 3: 1.5, 2: 1.25, 1: 1.0, 0: 0.75 }
else: # Default to Moderate
    alloc_map = { 4: 1.35, 3: 1.15, 2: 1.00, 1: 0.60, 0: 0.00 }

# Calculate daily base targets
qqq['Target_QQQ_Base'] = qqq['Regime'].map(alloc_map)

# --- APPLICATION OF VOLATILITY OVERRIDE ---
if enable_vol_override:
    qqq['Target_QQQ_Base'] = np.where(
        qqq['Vol_Extreme'], 
        np.minimum(qqq['Target_QQQ_Base'], vol_override_cap), 
        qqq['Target_QQQ_Base']
    )

qqq['Target_Hedge_Base'] = np.maximum(0, 1.0 - qqq['Target_QQQ_Base'])

# Apply Rebalancing Frequency
if rebalance_freq == "Weekly":
    weeks = qqq.index.to_series().dt.isocalendar().week
    is_end_of_week = weeks != weeks.shift(-1)
    
    qqq['Target_QQQ'] = qqq['Target_QQQ_Base'].where(is_end_of_week).ffill().bfill()
    qqq['Target_Hedge'] = qqq['Target_Hedge_Base'].where(is_end_of_week).ffill().bfill()
else:
    qqq['Target_QQQ'] = qqq['Target_QQQ_Base']
    qqq['Target_Hedge'] = qqq['Target_Hedge_Base']

# =====================
# 4. Portfolio Simulation
# =====================
qqq['Alloc_QQQ'] = qqq['Target_QQQ'].shift(1)
qqq['Alloc_Hedge'] = qqq['Target_Hedge'].shift(1)

qqq['Ret_QQQ'] = qqq['Close'].pct_change()
qqq['Ret_Hedge'] = hedge['Close'].pct_change()

qqq['Port_Ret'] = (qqq['Alloc_QQQ'] * qqq['Ret_QQQ']) + (qqq['Alloc_Hedge'] * qqq['Ret_Hedge'].fillna(0))

# Margin Borrowing Drag
margin_rate_annual = 0.0683 
qqq['Margin_Drag'] = np.maximum(0, qqq['Alloc_QQQ'] - 1.0) * (margin_rate_annual / 252)
qqq['Port_Ret'] = qqq['Port_Ret'] - qqq['Margin_Drag']

# Drop warmup period
qqq = qqq.dropna(subset=['MA_Slope', 'Vol_Mean']).copy()

# Capital Simulation
qqq['Portfolio_Value'] = initial_capital * (1 + qqq['Port_Ret']).cumprod()
qqq['BnH_Value'] = initial_capital * (1 + qqq['Ret_QQQ']).cumprod()

# Percentage Return for Y-Axis
qqq['Port_Return_Pct'] = (qqq['Portfolio_Value'] / initial_capital - 1) * 100
qqq['BnH_Return_Pct'] = (qqq['BnH_Value'] / initial_capital - 1) * 100

# =====================
# 5. Performance Metrics & Live Orders
# =====================
total_days = len(qqq)
years = total_days / 252

total_return_strat = (qqq['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
total_return_bnh = (qqq['BnH_Value'].iloc[-1] / initial_capital - 1) * 100

cagr = (qqq['Portfolio_Value'].iloc[-1] / initial_capital) ** (1 / years) - 1

rolling_max = qqq['Portfolio_Value'].cummax()
drawdown = (qqq['Portfolio_Value'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

risk_free_rate = 0.04 / 252
excess_returns = qqq['Port_Ret'] - risk_free_rate
sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

downside_returns = excess_returns.copy()
downside_returns[downside_returns > 0] = 0
downside_deviation = downside_returns.std()
sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252)

print("\n=== STRATEGY HISTORICAL PERFORMANCE ===")
print(f"Hedge Vehicle:      {hedge_ticker}")
print(f"Profile:            {allocation_profile}")
print(f"Rebalance Freq:     {rebalance_freq}")
print(f"MA Type:            {ma_type.upper()} ({ma_len}-Day)")
print(f"Vol Override:       {'ON' if enable_vol_override else 'OFF'} (Cap: {vol_override_cap}x)")
print(f"Starting Capital:   ${initial_capital:,.2f}")
print(f"Ending Capital:     ${qqq['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return (Strat):{total_return_strat:.2f}%")
print(f"Total Return (B&H): {total_return_bnh:.2f}%")
print(f"CAGR:               {cagr * 100:.2f}%")
print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
print(f"Sortino Ratio:      {sortino_ratio:.2f}")
print("========================================\n")

# --- LIVE REBALANCE LINK GENERATOR ---
live_regime = int(qqq['Regime'].iloc[-1])
current_price = qqq['Close'].iloc[-1]
hedge_price = hedge['Close'].iloc[-1]

base_url = "https://paulcwarren.github.io/pinescript-libs/strategies/qqq/rebal.html" 

print("\n=== REBALANCE LINK FOR THE FAMILY ===")
print(f"{base_url}?regime={live_regime}&qqq={current_price:.2f}&hedge={hedge_price:.2f}")
print("=====================================\n")

# =====================
# 6. Visualization
# =====================
plt.figure(figsize=(12, 6))
plt.plot(qqq.index, qqq['Port_Return_Pct'], label=f'Adaptive Strategy ({allocation_profile})', color='blue', linewidth=2)
plt.plot(qqq.index, qqq['BnH_Return_Pct'], label='Buy & Hold QQQ', color='orange', alpha=0.7)
plt.title(f'Regime Strategy Comparison (2020 - Present)')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()