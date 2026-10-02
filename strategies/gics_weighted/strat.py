import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class GICSWeightedStrategy:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        
        self.tickers = [
            "XLK", "XLV", "XLF", "XLY", "XLC", 
            "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB"
        ]
        self.benchmark_ticker = "SPY"
        
        self.data = None
        self.benchmark_data = None
        self.equity_curve = None
        self.benchmark_curve = None
        self.initial_capital = 10000
        
        # Approximate SPY Cap-Weights
        self.cap_weights = {
            "XLK": 0.300, "XLF": 0.130, "XLV": 0.120, "XLY": 0.100, 
            "XLC": 0.090, "XLI": 0.090, "XLP": 0.060, "XLE": 0.040, 
            "XLU": 0.025, "XLRE": 0.025, "XLB": 0.020
        }

    def fetch_data(self):
        print("Fetching OHLC data for strategy and benchmark...")
        
        # Fetch strategy data (Need High, Low, Close for swing logic)
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        self.data = df[['Close', 'High', 'Low']].ffill().dropna()
        
        # Fetch benchmark data
        bench_df = yf.download(self.benchmark_ticker, start=self.start_date, end=self.end_date)["Close"]
        self.benchmark_data = bench_df.squeeze().reindex(self.data.index).ffill()
        
        print("Data fetched successfully.\n")

    def run_backtest(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")

        close_prices = self.data['Close']
        high_prices = self.data['High']
        low_prices = self.data['Low']
        
        daily_returns = close_prices.pct_change().fillna(0)
        dates = close_prices.index
        
        # Calculate Indicators
        sma12 = close_prices.rolling(12).mean()
        sma22 = close_prices.rolling(22).mean()
        sma55 = close_prices.rolling(55).mean()
        
        consec_gt_55 = (close_prices > sma55).rolling(5).sum() == 5
        consec_lt_55 = (close_prices < sma55).rolling(5).sum() == 5

        # States: 0 = Underweight (UW), 1 = Equal Weight (EW), 2 = Overweight (OW)
        states = {ticker: 1 for ticker in self.tickers}
        entry_reasons = {ticker: 'init' for ticker in self.tickers}
        swing_lows = {ticker: 0.0 for ticker in self.tickers}
        swing_highs = {ticker: 0.0 for ticker in self.tickers}
        
        portfolio_returns = []
        
        for i in range(len(dates)):
            if i < 55:
                daily_port_ret = sum(daily_returns[t].iloc[i] * self.cap_weights[t] for t in self.tickers)
                portfolio_returns.append(daily_port_ret)
                continue
                
            # 1. Evaluate State Transitions
            for t in self.tickers:
                c = close_prices[t].iloc[i]
                prev_c = close_prices[t].iloc[i-1]
                s12, p_s12 = sma12[t].iloc[i], sma12[t].iloc[i-1]
                s22, p_s22 = sma22[t].iloc[i], sma22[t].iloc[i-1]
                s55 = sma55[t].iloc[i]
                
                if consec_gt_55[t].iloc[i]:
                    states[t] = 2
                    entry_reasons[t] = '5_gt'
                elif consec_lt_55[t].iloc[i]:
                    states[t] = 0
                    entry_reasons[t] = '5_lt'
                else:
                    curr_state = states[t]
                    if curr_state == 0:
                        if (p_s12 <= p_s22) and (s12 > s22) and (s12 < s55) and (s22 < s55):
                            states[t] = 1
                            entry_reasons[t] = 'cross_up'
                            swing_lows[t] = low_prices[t].iloc[i-10:i+1].min()
                    elif curr_state == 2:
                        if (p_s12 >= p_s22) and (s12 < s22) and (s12 > s55) and (s22 > s55):
                            states[t] = 1
                            entry_reasons[t] = 'cross_down'
                            swing_highs[t] = high_prices[t].iloc[i-10:i+1].max()
                    elif curr_state == 1:
                        if entry_reasons[t] == 'cross_up' and c < swing_lows[t]:
                            states[t] = 0
                            entry_reasons[t] = 'fail_low'
                        elif entry_reasons[t] == 'cross_down' and c > swing_highs[t]:
                            states[t] = 2
                            entry_reasons[t] = 'fail_high'

            # 2. Calculate Allocation Weights (UPDATED LOGIC)
            uw_cash_pool = sum(self.cap_weights[t] for t in self.tickers if states[t] == 0)
            ow_count = sum(1 for t in self.tickers if states[t] == 2) # Count how many sectors are Overweight
            
            day_return = 0.0
            
            if i < len(dates) - 1:
                for t in self.tickers:
                    target_weight = 0.0
                    if states[t] == 1:
                        target_weight = self.cap_weights[t]
                    elif states[t] == 2:
                        if ow_count > 0:
                            # Base weight + EQUAL share of the Underweight cash pool
                            bonus = uw_cash_pool / ow_count 
                            target_weight = self.cap_weights[t] + bonus
                        else:
                            target_weight = self.cap_weights[t]
                    
                    day_return += target_weight * daily_returns[t].iloc[i+1]
                    
            portfolio_returns.append(day_return)

        portfolio_returns = [0] + portfolio_returns[:-1]
        strat_returns = pd.Series(portfolio_returns, index=dates)

        self.equity_curve = self.initial_capital * (1 + strat_returns).cumprod()
        
        bench_returns = self.benchmark_data.pct_change().dropna()
        self.benchmark_curve = self.initial_capital * (1 + bench_returns).cumprod()
        start_val = pd.Series([self.initial_capital], index=[self.benchmark_data.index[0]])
        self.benchmark_curve = pd.concat([start_val, self.benchmark_curve])

    def print_summary(self):
        if self.equity_curve is None:
            return

        def calc_metrics(curve):
            if isinstance(curve, pd.DataFrame):
                curve = curve.squeeze()
                
            returns = curve.pct_change().dropna()
            total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
            
            days = (curve.index[-1] - curve.index[0]).days
            cagr = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
            
            volatility = float(returns.std() * np.sqrt(252))
            sharpe = cagr / volatility if volatility != 0 else 0
            
            rolling_max = curve.cummax()
            drawdown = (curve / rolling_max) - 1
            max_drawdown = float(drawdown.min())
            
            return total_return, cagr, volatility, sharpe, max_drawdown

        strat_tr, strat_cagr, strat_vol, strat_sharpe, strat_dd = calc_metrics(self.equity_curve)
        spy_tr, spy_cagr, spy_vol, spy_sharpe, spy_dd = calc_metrics(self.benchmark_curve)

        print("="*55)
        print(f"{'PERFORMANCE SUMMARY':^55}")
        print("="*55)
        print(f"{'Metric':<20} | {'Strategy':<14} | {'SPY Benchmark':<14}")
        print("-" * 55)
        print(f"{'Total Return':<20} | {strat_tr:>13.2%} | {spy_tr:>13.2%}")
        print(f"{'Annualized (CAGR)':<20} | {strat_cagr:>13.2%} | {spy_cagr:>13.2%}")
        print(f"{'Annual Volatility':<20} | {strat_vol:>13.2%} | {spy_vol:>13.2%}")
        print(f"{'Sharpe Ratio':<20} | {strat_sharpe:>13.2f} | {spy_sharpe:>13.2f}")
        print(f"{'Max Drawdown':<20} | {strat_dd:>13.2%} | {spy_dd:>13.2%}")
        print("="*55 + "\n")

    def plot_results(self):
        if self.equity_curve is None:
            raise ValueError("No equity curve available. Call run_backtest() first.")
            
        plt.figure(figsize=(12, 6))
        
        plt.plot(
            self.equity_curve.index, 
            self.equity_curve.values, 
            label='Rotational Strategy', 
            color='#1f77b4', 
            linewidth=2
        )
        
        plt.plot(
            self.benchmark_curve.index, 
            self.benchmark_curve.values, 
            label='SPY Benchmark', 
            color='#7f7f7f', 
            linestyle='--', 
            linewidth=1.5
        )
        
        plt.title('Rotational Sector Strategy vs. SPY Benchmark', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    strategy = GICSWeightedStrategy(start_date="2020-01-01", end_date="2024-01-01")
    strategy.fetch_data()
    strategy.run_backtest()
    strategy.print_summary()
    strategy.plot_results()