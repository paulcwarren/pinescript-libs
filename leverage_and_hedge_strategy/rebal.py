import yfinance as yf
import math
import warnings

# Suppress yfinance warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# 1. YOUR WEEKLY INPUTS (UPDATE THESE ON FRIDAYS)
# ==========================================
strategy_capital = 50000      # The dollar amount dedicated to this strategy
current_qqq_shares = 100      # How many QQQ shares you currently hold
# ==========================================

print("Fetching live market data for QQQ...\n")
# Fetch the last 6 months of daily data
data = yf.download("QQQ", period="6mo", progress=False)

# Handle yfinance multi-index columns if present
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close']['QQQ']
else:
    close_prices = data['Close']

# ==========================================
# 2. CALCULATE INDICATORS & REGIME
# ==========================================
current_price = close_prices.iloc[-1]

# Calculate the 100-Day SMA
sma_100 = close_prices.rolling(window=100).mean()
current_sma = sma_100.iloc[-1]
past_sma = sma_100.iloc[-6]  # Look back 5 days to determine slope

# Logic Checks
price_above_sma = current_price > current_sma
slope_up = current_sma > past_sma

# Determine Regime (Using the Moderate Profile)
if price_above_sma and slope_up:
    regime = 4
    target_leverage = 1.35
elif price_above_sma and not slope_up:
    regime = 3
    target_leverage = 1.15
elif not price_above_sma and slope_up:
    regime = 2
    target_leverage = 1.00
else: 
    # Price is below SMA and slope is down
    regime = 1
    target_leverage = 0.60

# ==========================================
# 3. POSITION MATH
# ==========================================
target_position_value = strategy_capital * target_leverage

# We use math.floor to round down to the nearest whole share. 
# This prevents margin creep from fractional rounding.
target_shares = math.floor(target_position_value / current_price)
shares_to_trade = target_shares - current_qqq_shares

# ==========================================
# 4. FIDELITY TRADE OUTPUT
# ==========================================
print("-" * 50)
print(f"FRIDAY REBALANCE SUMMARY")
print("-" * 50)
print(f"Current QQQ Price:  ${current_price:.2f}")
print(f"100-Day SMA:        ${current_sma:.2f}")
print(f"SMA Slope is:       {'RISING' if slope_up else 'FALLING'}")
print(f"Current Regime:     Regime {regime}")
print(f"Target Leverage:    {target_leverage}x")
print("-" * 50)

if shares_to_trade > 0:
    print(f"🔥 ACTION REQUIRED: BUY {shares_to_trade} SHARES OF QQQ")
elif shares_to_trade < 0:
    print(f"🛡️ ACTION REQUIRED: SELL {abs(shares_to_trade)} SHARES OF QQQ")
else:
    print("✅ NO ACTION REQUIRED: You are perfectly balanced.")
print("-" * 50)