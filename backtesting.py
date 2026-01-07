"""
Backtesting regime-based trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RegimeStrategy:
    """
    Base class for regime-based trading strategies.
    
    Strategy logic: Use detected anomalies as signals for portfolio adjustment.
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
    
    def backtest(self, prices, distance_df, strategy='avoid_anomalies'):
        """
        Backtest strategy using regime signals.
        
        Strategies:
        -----------
        - 'avoid_anomalies': Exit positions during detected anomalies
        - 'buy_dips': Buy during anomalies (contrarian)
        - 'dynamic_hedge': Reduce exposure proportional to W₁ distance
        """
        # Align prices with distance_df
        common_dates = prices.index.intersection(distance_df.index)
        prices = prices.loc[common_dates]
        distance_df = distance_df.loc[common_dates]
        
        if strategy == 'avoid_anomalies':
            returns = self._avoid_anomalies(prices, distance_df)
        elif strategy == 'buy_dips':
            returns = self._buy_dips(prices, distance_df)
        elif strategy == 'dynamic_hedge':
            returns = self._dynamic_hedge(prices, distance_df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Compute performance metrics
        self.results = self._compute_metrics(returns, prices)
        
        return self.results
    
    def _avoid_anomalies(self, prices, distance_df):
        """
        Strategy 1: Hold 100% during normal, 0% during anomalies.
        """
        position = np.ones(len(prices))
        position[distance_df['anomaly']] = 0.0
        
        price_returns = prices.pct_change()
        strategy_returns = position * price_returns
        
        self.positions = pd.DataFrame({
            'date': prices.index,
            'position': position,
            'price': prices.values
        }).set_index('date')
        
        return strategy_returns
    
    def _buy_dips(self, prices, distance_df):
        """
        Strategy 2: Increase position during anomalies (contrarian).
        """
        position = np.ones(len(prices))
        position[distance_df['anomaly']] = 1.5  # 50% overweight
        
        price_returns = prices.pct_change()
        strategy_returns = position * price_returns
        
        self.positions = pd.DataFrame({
            'date': prices.index,
            'position': position,
            'price': prices.values
        }).set_index('date')
        
        return strategy_returns
    
    def _dynamic_hedge(self, prices, distance_df):
        """
        Strategy 3: Position size inversely proportional to W₁ distance.
        
        position = max(0, 1 - k * W₁) where k is sensitivity
        """
        # Normalize distances
        W_max = distance_df['wasserstein'].quantile(0.95)
        W_norm = distance_df['wasserstein'] / W_max
        
        # Position: 100% when W=0, 0% when W≥W_max
        position = np.clip(1.0 - W_norm, 0.0, 1.0)
        
        price_returns = prices.pct_change()
        strategy_returns = position.values * price_returns
        
        self.positions = pd.DataFrame({
            'date': prices.index,
            'position': position.values,
            'price': prices.values
        }).set_index('date')
        
        return strategy_returns
    
    def _compute_metrics(self, strategy_returns, prices):
        """
        Compute performance metrics.
        """
        # Buy-and-hold baseline
        bh_returns = prices.pct_change()
        
        # Cumulative returns
        strat_cum = (1 + strategy_returns).cumprod()
        bh_cum = (1 + bh_returns).cumprod()
        
        # Annualized metrics (assume 252 trading days)
        n_years = len(strategy_returns) / 252
        
        strat_total = strat_cum.iloc[-1] - 1
        bh_total = bh_cum.iloc[-1] - 1
        
        strat_annual = (1 + strat_total) ** (1/n_years) - 1
        bh_annual = (1 + bh_total) ** (1/n_years) - 1
        
        # Volatility
        strat_vol = strategy_returns.std() * np.sqrt(252)
        bh_vol = bh_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assume 0% risk-free rate)
        strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
        bh_sharpe = bh_annual / bh_vol if bh_vol > 0 else 0
        
        # Max drawdown
        strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
        bh_dd = (bh_cum / bh_cum.cummax() - 1).min()
        
        results = {
            'strategy_return': strat_total,
            'buyhold_return': bh_total,
            'strategy_annual': strat_annual,
            'buyhold_annual': bh_annual,
            'strategy_vol': strat_vol,
            'buyhold_vol': bh_vol,
            'strategy_sharpe': strat_sharpe,
            'buyhold_sharpe': bh_sharpe,
            'strategy_maxdd': strat_dd,
            'buyhold_maxdd': bh_dd,
            'strategy_cumulative': strat_cum,
            'buyhold_cumulative': bh_cum
        }
        
        return results
    
    def plot_performance(self):
        """
        Plot cumulative returns comparison.
        """
        if not hasattr(self, 'results'):
            raise ValueError("Must run backtest() first")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Cumulative returns
        ax = axes[0]
        strat_cum = self.results['strategy_cumulative']
        bh_cum = self.results['buyhold_cumulative']
        
        ax.plot(strat_cum.index, strat_cum, 
                color='darkgreen', linewidth=2, label='Strategy')
        ax.plot(bh_cum.index, bh_cum,
                color='gray', linewidth=2, alpha=0.7, label='Buy & Hold')
        
        ax.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
        ax.set_title('Regime Strategy Performance', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Position sizing
        ax = axes[1]
        ax.plot(self.positions.index, self.positions['position'],
                color='steelblue', linewidth=1.5, label='Position Size')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
        
        ax.set_ylabel('Position Size', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylim([-0.1, 1.6])
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """
        Print backtest summary.
        """
        if not hasattr(self, 'results'):
            raise ValueError("Must run backtest() first")
        
        r = self.results
        
        print("="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        print(f"\n{'Metric':<30} {'Strategy':>15} {'Buy & Hold':>15}")
        print("-"*70)
        print(f"{'Total Return':<30} {r['strategy_return']:>14.2%} {r['buyhold_return']:>14.2%}")
        print(f"{'Annualized Return':<30} {r['strategy_annual']:>14.2%} {r['buyhold_annual']:>14.2%}")
        print(f"{'Volatility':<30} {r['strategy_vol']:>14.2%} {r['buyhold_vol']:>14.2%}")
        print(f"{'Sharpe Ratio':<30} {r['strategy_sharpe']:>15.2f} {r['buyhold_sharpe']:>15.2f}")
        print(f"{'Max Drawdown':<30} {r['strategy_maxdd']:>14.2%} {r['buyhold_maxdd']:>14.2%}")
        print("="*70)
        
        # Relative performance
        excess_return = r['strategy_annual'] - r['buyhold_annual']
        vol_reduction = (r['buyhold_vol'] - r['strategy_vol']) / r['buyhold_vol']
        
        print(f"\nExcess Return: {excess_return:+.2%}")
        print(f"Volatility Reduction: {vol_reduction:.1%}")
        print(f"Risk-Adjusted Improvement: {r['strategy_sharpe'] - r['buyhold_sharpe']:+.2f} Sharpe")


if __name__ == "__main__":
    import yfinance as yf
    from regime_detector import RegimeDetector
    
    print("="*80)
    print("REGIME STRATEGY BACKTESTING")
    print("="*80)
    
    # Data
    spy = yf.download('SPY', start='2018-01-01', end='2024-01-01', progress=False)
    
    # Train detector
    print("\nTraining detector...")
    detector = RegimeDetector(window=60, reference_period=252, contamination=0.1)
    detector.fit(spy['Adj Close'], spy['Volume'], end_date='2021-12-31')
    
    # Out-of-sample test
    print("Running out-of-sample predictions...")
    test_predictions = detector.predict(
        spy['Adj Close'].loc['2022-01-01':],
        spy['Volume'].loc['2022-01-01':]
    )
    
    # Backtest strategies
    strategies = ['avoid_anomalies', 'buy_dips', 'dynamic_hedge']
    
    for strat_name in strategies:
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strat_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        strategy = RegimeStrategy()
        strategy.backtest(
            spy['Adj Close'].loc['2022-01-01':],
            test_predictions,
            strategy=strat_name
        )
        
        strategy.print_summary()
        
        # Plot
        fig = strategy.plot_performance()
        plt.savefig(f'backtest_{strat_name}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved backtest_{strat_name}.png")
    
    print(f"\n{'='*80}")
    print("✓ Backtesting complete.")
    print(f"{'='*80}")
    plt.show()
