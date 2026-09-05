import os
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

# Initialize FRED API (automatically pulls FRED_API_KEY from your direnv setup)
fred = Fred(api_key=os.environ.get('FRED_API_KEY'))

print("Fetching Z.1 Flow of Funds balance sheet data from FRED...")

try:
    # Pulling direct FRED Z.1 series
    equities        = fred.get_series('BOGZ1FL153064486Q') # Directly & Indirectly Held Equities (% of Financial Assets)
    tot_fin_assets  = fred.get_series('TFAABSHNO')         # Total Financial Assets ($ Millions)
    tot_assets      = fred.get_series('TABSHNO')           # Total Assets ($ Millions)
    checkable_dep   = fred.get_series('CDCABSHNO')          # Checkable Deposits & Currency ($ Millions)
    mmf_shares      = fred.get_series('MMFSABSHNO')         # Money Market Fund Shares ($ Millions)
    liabilities     = fred.get_series('TLBSHNO')            # Total Liabilities ($ Millions)
    net_worth       = fred.get_series('TNWBSHNO')           # Net Worth ($ Millions)

    # Combine into a clean DataFrame using inner alignment across series
    df = pd.DataFrame({
        'Equity_Allocation_Fin_Pct': equities,
        'Total_Financial_Assets': tot_fin_assets,
        'Total_Assets': tot_assets,
        'Checkable_Deposits': checkable_dep,
        'MMF_Shares': mmf_shares,
        'Liabilities': liabilities,
        'Net_Worth': net_worth
    }).dropna()

    # Convert everything to a shared % of Total Assets denominator:
    # 1. Equities (originally % of financial assets) -> % of total assets
    df['Equities_Total_Assets_Pct'] = (df['Equity_Allocation_Fin_Pct'] / 100) * (df['Total_Financial_Assets'] / df['Total_Assets']) * 100

    # 2. Cash & MMFs -> % of total assets
    df['Cash_And_MMF'] = df['Checkable_Deposits'] + df['MMF_Shares']
    df['Cash_MMF_Total_Assets_Pct'] = (df['Cash_And_MMF'] / df['Total_Assets']) * 100

    # 3. Real Estate / Nonfinancial Assets -> % of total assets
    df['Nonfinancial_Assets'] = df['Total_Assets'] - df['Total_Financial_Assets']
    df['Real_Estate_Share_Pct'] = (df['Nonfinancial_Assets'] / df['Total_Assets']) * 100

    # 4. Leverage Ratio
    df['Debt_To_Net_Worth_Pct'] = (df['Liabilities'] / df['Net_Worth']) * 100

    # Plotting configuration (2 stacked subplots: Universal % of Total Assets + Leverage)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Universal Asset Composition (% of Total Assets)
    axes[0].plot(df.index, df['Equities_Total_Assets_Pct'], label='Corporate Equities (% Total Assets)', color='#1f77b4', lw=2)
    axes[0].plot(df.index, df['Real_Estate_Share_Pct'], label='Real Estate & Tangibles (% Total Assets)', color='#ff7f0e', lw=2)
    axes[0].plot(df.index, df['Cash_MMF_Total_Assets_Pct'], label='Cash & MMFs (% Total Assets)', color='#2ca02c', lw=2)
    
    axes[0].set_title('U.S. Household Macro Asset Composition (% of Total Assets)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Share of Total Assets (%)', fontsize=11)
    axes[0].legend(loc='upper left', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Household Leverage (Debt-to-Net Worth)
    axes[1].plot(df.index, df['Debt_To_Net_Worth_Pct'], label='Household Debt / Net Worth', color='#d62728', lw=2)
    axes[1].set_title('U.S. Household Leverage: Debt-to-Net Worth Ratio', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Ratio (%)', fontsize=11)
    axes[1].set_xlabel('Quarter', fontsize=11)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('household_z1_analysis.png', dpi=300)
    print("Plot successfully saved as 'household_z1_analysis.png'")
    plt.show()

except Exception as e:
    print("Error fetching or processing Z.1 FRED series:", e)