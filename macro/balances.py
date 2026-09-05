import os
import argparse
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Plot US Sectoral Balances')
parser.add_argument('--detailed', action='store_true', help='Split the Private sector into Household and Corporate balances')
args = parser.parse_args()

# Retrieve API key from direnv
api_key = os.environ.get('FRED_API_KEY')
if not api_key:
    raise ValueError("FRED_API_KEY environment variable is not set. Please set it before running.")

# Set up FRED API
fred = Fred(api_key=api_key)

# 1. Fetch Official Quarterly NIPA Data (Natively in Billions of $)
private_total = fred.get_series('W994RC1Q027SBEA')
gov           = fred.get_series('AD01RC1Q027SBEA')
foreign       = -fred.get_series('NETFI') # Invert US current account
household     = fred.get_series('W995RC1Q027SBEA') # Households and nonprofits

# 2. Construct DataFrame (Convert Billions to Trillions of $)
balances = pd.DataFrame({
    'Private': private_total / 1000,
    'Household': household / 1000,
    'Gov': gov / 1000,
    'Foreign': foreign / 1000
}).dropna()

# Derive Corporate balance (Total Private - Household)
balances['Corporate'] = balances['Private'] - balances['Household']

# Calculate the True Identity Net Total
balances['Net Total'] = balances['Private'] + balances['Gov'] + balances['Foreign']

# 3. Plotting
plt.figure(figsize=(12, 6))

# Toggle between detailed and simple views based on the program flag
if args.detailed:
    plt.plot(balances.index, balances['Household'], label='Household Balance', color='dodgerblue', linewidth=1.5)
    plt.plot(balances.index, balances['Corporate'], label='Corporate Balance', color='navy', linewidth=1.5)
    plt.title('Sectoral Balances (Trillions of USD) - Detailed View')
else:
    plt.plot(balances.index, balances['Private'], label='Private Balance', color='blue', linewidth=1.5)
    plt.title('Sectoral Balances (Trillions of USD)')

plt.plot(balances.index, balances['Gov'], label='Gov Balance', color='red', linewidth=1.5)
plt.plot(balances.index, balances['Foreign'], label='Foreign (Trade) Balance', color='green', linewidth=1.5)
plt.plot(balances.index, balances['Net Total'], label='Net Total', color='purple', linestyle='--', linewidth=2)

# Formatting the chart
plt.axhline(0, color='black', linestyle='-', linewidth=1)
plt.ylabel('Trillions of USD')
plt.xlabel('Year')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Display the chart
plt.show()