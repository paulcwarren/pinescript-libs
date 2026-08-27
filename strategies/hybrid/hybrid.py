import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings('ignore')

# =====================
# 1. Configuration
# =====================
start_date = "2020-12-02" 
primary_ticker = "QQQ"
hedge_ticker = "DBMF"
initial_capital = 10000

# Strategy Params
target_vol = 0.25
max_leverage = 2.0
rebalance_freq = "Weekly"
trade_mode = "UTMA_BLEND" 

print("Downloading Hybrid Strategy data...")
tickers = [primary_ticker, hedge_ticker, 'TQQQ']
data = yf.download(tickers, start=start_date, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    qqq = data.xs(primary_ticker, level=1, axis=1).copy()
    hedge = data.xs(hedge_ticker, level=1, axis=1).copy()
    tqqq = data.xs('TQQQ', level=1, axis=1).copy()
else:
    qqq = data[[primary_ticker]].copy()
    hedge = data[[hedge_ticker]].copy()
    tqqq = data[['TQQQ']].copy()

# =====================
# 2. Indicators & Regimes
# =====================
ma_len = 100
slope_lookback = 5
atr_len = 14
vol_len = 50

# Moving Average
if "EMA" == "SMA": # Using SMA based on your moderate profile
    pass
qqq['MA'] = qqq['Close'].rolling(window=ma_len).mean()
qqq['MA_Slope'] = (qqq['MA'] - qqq['MA'].shift(slope_lookback)) / qqq['MA'].shift(slope_lookback)

# ATR & Volatility Expansion
tr1 = qqq['High'] - qqq['Low']
tr2 = (qqq['High'] - qqq['Close'].shift(1)).abs()
tr3 = (qqq['Low'] - qqq['Close'].shift(1)).abs()
qqq['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
qqq['ATR'] = qqq['TR'].ewm(alpha=1/atr_len, adjust=False).mean()
qqq['Vol'] = qqq['ATR'] / qqq['Close']
qqq['Vol_Mean'] = qqq['Vol'].rolling(window=vol_len).mean()
qqq['Vol_Expand'] = qqq['Vol'] > qqq['Vol_Mean'] 

# Determine Regimes
cond_close_above = qqq['Close'] > qqq['MA']
cond_slope_up = qqq['MA_Slope'] > 0

qqq['Regime'] = 1 
qqq.loc[~cond_close_above & cond_slope_up, 'Regime'] = 2  
qqq.loc[cond_close_above & ~cond_slope_up, 'Regime'] = 3  
qqq.loc[cond_close_above & cond_slope_up, 'Regime'] = 4   
qqq.loc[~cond_close_above & ~cond_slope_up & qqq['Vol_Expand'], 'Regime'] = 0  

# =====================
# 3. Volatility Targeting Logic
# =====================
daily_ret = qqq['Close'].pct_change().fillna(0)
rolling_vol = daily_ret.rolling(window=20).std() * np.sqrt(252)

# Volatility Target / Current Volatility, capped at 2.0x
vol_leverage = (target_vol / rolling_vol).clip(upper=max_leverage)
qqq['Vol_Lev'] = vol_leverage.shift(1).fillna(1.0)

# =====================
# 4. Hybrid Target Allocation
# =====================
def get_hybrid_target(row):
    r = row['Regime']
    # If in Regime 4 (Uptrend), let the vol-target dictate leverage
    if r == 4:
        return row['Vol_Lev']
    # Defensive postures for other regimes (Moderate Profile values)
    elif r == 3: return 1.15
    elif r == 2: return 1.00
    elif r == 1: return 0.60
    elif r == 0: return 0.00
    return 1.0

qqq['Target_QQQ_Base'] = qqq.apply(get_hybrid_target, axis=1)
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
# 5. Portfolio Simulation (UTMA BLEND)
# =====================
qqq['Ret_QQQ'] = qqq['Close'].pct_change()
qqq['Ret_Hedge'] = hedge['Close'].pct_change()
qqq['Ret_TQQQ'] = tqqq['Close'].pct_change()

qqq['Alloc_QQQ'] = qqq['Target_QQQ'].shift(1)
qqq['Alloc_Hedge'] = qqq['Target_Hedge'].shift(1)

# Blend Mode (Zero margin cost, blending cash QQQ & TQQQ)
qqq['W_TQQQ'] = np.maximum(0, (qqq['Alloc_QQQ'] - 1.0) / 2.0)
qqq['W_QQQ'] = np.where(qqq['Alloc_QQQ'] > 1.0, 1.0 - qqq['W_TQQQ'], qqq['Alloc_QQQ'])

qqq['Port_Ret'] = (qqq['W_QQQ'] * qqq['Ret_QQQ']) + \
                  (qqq['W_TQQQ'] * qqq['Ret_TQQQ'].fillna(0)) + \
                  (qqq['Alloc_Hedge'] * qqq['Ret_Hedge'].fillna(0))

qqq = qqq.dropna(subset=['MA_Slope', 'Vol_Mean']).copy()

qqq['Portfolio_Value'] = initial_capital * (1 + qqq['Port_Ret']).cumprod()
qqq_bnh = initial_capital * (1 + qqq['Ret_QQQ']).cumprod()

# --- Performance Metrics ---
total_days = len(qqq)
years = total_days / 252

total_return_strat = (qqq['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
total_return_bnh = (qqq_bnh.iloc[-1] / initial_capital - 1) * 100

cagr = (qqq['Portfolio_Value'].iloc[-1] / initial_capital) ** (1 / years) - 1

rolling_max = qqq['Portfolio_Value'].cummax()
max_drawdown = ((qqq['Portfolio_Value'] - rolling_max) / rolling_max).min()

bnh_rolling_max = qqq_bnh.cummax()
bnh_max_drawdown = ((qqq_bnh - bnh_rolling_max) / bnh_rolling_max).min()

risk_free = 0.04 / 252
excess_ret = qqq['Port_Ret'] - risk_free
sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)

downside = excess_ret.copy()
downside[downside > 0] = 0
sortino = (excess_ret.mean() / downside.std()) * np.sqrt(252)

print("\n=== STRATEGY HISTORICAL PERFORMANCE ===")
print(f"Strategy:           Regime-Filtered Volatility Targeting")
print(f"Underlying:         {primary_ticker}")
print(f"Target Volatility:  {target_vol * 100:.0f}% Annualized")
print(f"Max Leverage:       {max_leverage}x (UTMA Blend Mode)")
print(f"Starting Capital:   ${initial_capital:,.2f}")
print(f"Ending Capital:     ${qqq['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return (Vol): {total_return_strat:.2f}%")
print(f"Total Return (QQQ): {total_return_bnh:.2f}%")
print(f"CAGR:               {cagr * 100:.2f}%")
print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
print(f"Sharpe Ratio:       {sharpe:.2f}")
print(f"Sortino Ratio:      {sortino:.2f}")
print("========================================\n")

# --- LIVE REBALANCE LINK ---
live_regime = int(qqq['Regime'].iloc[-1])
live_leverage = float(qqq['Target_QQQ'].iloc[-1])
current_price = float(qqq['Close'].iloc[-1])
hedge_price = float(hedge['Close'].iloc[-1])
tqqq_price = float(tqqq['Close'].iloc[-1])

base_url = "https://paulcwarren.github.io/pinescript-libs/strategies/hybrid/rebal.html" 
final_link = f"{base_url}?regime={live_regime}&lev={live_leverage:.2f}&qqq={current_price:.2f}&hedge={hedge_price:.2f}&hticker={hedge_ticker}&tqqq={tqqq_price:.2f}"

print("=== COPY AND PASTE THIS INTO WHATSAPP ===")
print("Yo fam! Wadup...It's time to rebalance your portfolios and keep making the wonga.  Here is your rebalance link:")
print()
print(final_link)
print("=========================================\n")

# --- Chart ---
plt.figure(figsize=(12, 6))
plt.plot(qqq.index, qqq['Portfolio_Value'], label='Hybrid Regime + Vol Target', color='purple', linewidth=2)
plt.plot(qqq.index, qqq_bnh, label='Buy & Hold QQQ', color='orange', alpha=0.7)

plt.title("Hybrid Strategy: Vol Targeting in Bull Regimes vs QQQ (Log Scale)")
plt.yscale('log')
plt.ylabel("Portfolio Value ($)")
dollar_formatter = ticker.StrMethodFormatter('${x:,.0f}')
plt.gca().yaxis.set_major_formatter(dollar_formatter)
plt.gca().yaxis.set_minor_formatter(dollar_formatter)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()