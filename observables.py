"""
Observable map construction for financial time series.
Translates from physics framework to market indicators.
"""

import numpy as np
import pandas as pd


def compute_returns(prices, method='log'):
    """
    Compute returns from price series.
    
    Parameters:
    -----------
    prices : pd.Series or pd.DataFrame
        Price data
    method : str
        'log' for log returns, 'simple' for arithmetic
    
    Returns:
    --------
    pd.Series or pd.DataFrame : Returns
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def realized_volatility(returns, window=20, annualize=True):
    """
    Rolling realized volatility.
    
    Physics analogue: Var(∇h) - variance of gradients
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def skewness(returns, window=60):
    """
    Rolling skewness of returns.
    
    Physics analogue: Third moment of height fluctuations
    Detects asymmetry (crashes vs rallies)
    """
    return returns.rolling(window).skew()


def kurtosis(returns, window=60):
    """
    Rolling kurtosis (excess).
    
    Physics analogue: Fourth moment
    Detects fat tails / extreme events
    """
    return returns.rolling(window).kurt()


def autocorrelation(returns, lag=1, window=60):
    """
    Rolling autocorrelation at given lag.
    
    Physics analogue: Temporal correlation function
    Measures momentum / mean reversion
    """
    def rolling_autocorr(x):
        if len(x) < lag + 1:
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    
    return returns.rolling(window).apply(rolling_autocorr, raw=True)


def hurst_exponent(returns, window=252):
    """
    Rolling Hurst exponent via R/S analysis.
    
    Physics analogue: Roughness exponent α
    H ~ 0.5: random walk
    H > 0.5: trending
    H < 0.5: mean-reverting
    """
    def compute_hurst(ts):
        if len(ts) < 20:
            return np.nan
        
        lags = range(2, min(20, len(ts) // 2))
        tau = []
        
        for lag in lags:
            std = np.std(ts)
            if std == 0:
                continue
            
            # Subtract mean
            ts_mean = ts - np.mean(ts)
            
            # Cumulative sum
            Z = np.cumsum(ts_mean)
            
            # Range
            R = np.max(Z) - np.min(Z)
            
            # Rescaled range
            RS = R / std if std > 0 else 0
            tau.append(RS)
        
        if len(tau) == 0:
            return np.nan
        
        # Log-log regression
        lags_array = np.array(list(range(2, 2 + len(tau))))
        tau_array = np.array(tau)
        
        # Filter out zeros/nans
        mask = (tau_array > 0) & np.isfinite(tau_array)
        if mask.sum() < 3:
            return np.nan
        
        log_lags = np.log(lags_array[mask])
        log_tau = np.log(tau_array[mask])
        
        poly = np.polyfit(log_lags, log_tau, 1)
        return poly[0]
    
    return returns.rolling(window).apply(compute_hurst, raw=False)


def max_drawdown(prices, window=252):
    """
    Rolling maximum drawdown.
    
    Physics analogue: Largest excursion from maximum
    Key risk metric for regime classification
    """
    def compute_dd(p):
        if len(p) == 0:
            return np.nan
        cummax = pd.Series(p).cummax()
        dd = (pd.Series(p) - cummax) / cummax
        return dd.min()
    
    return prices.rolling(window).apply(compute_dd, raw=False)


def volume_momentum(volume, window=20):
    """
    Rolling volume trend.
    
    Captures liquidity regime changes
    """
    return volume.rolling(window).mean() / volume.rolling(window*2).mean()


def observable_map(prices, volume=None, window=60):
    """
    Construct full observable map Φ: price trajectory → R^d.
    
    This is the key translation from physics to finance:
    - Physics: Φ(h) extracts spatial/temporal statistics
    - Finance: Φ(S) extracts return distribution features
    
    Parameters:
    -----------
    prices : pd.Series
        Price time series
    volume : pd.Series, optional
        Volume time series
    window : int
        Lookback window for rolling statistics
    
    Returns:
    --------
    pd.DataFrame : Observable features
    """
    returns = compute_returns(prices, method='log')
    
    features = pd.DataFrame(index=prices.index)
    
    # First-order: moments of return distribution
    features['volatility'] = realized_volatility(returns, window)
    features['skewness'] = skewness(returns, window)
    features['kurtosis'] = kurtosis(returns, window)
    
    # Second-order: temporal structure
    features['autocorr_1'] = autocorrelation(returns, lag=1, window=window)
    features['autocorr_5'] = autocorrelation(returns, lag=5, window=window)
    
    # Scaling properties
    features['hurst'] = hurst_exponent(returns, window)
    
    # Risk measures
    features['max_drawdown'] = max_drawdown(prices, window)
    features['returns_mean'] = returns.rolling(window).mean() * 252  # Annualized
    
    # Tail risk
    features['var_95'] = returns.rolling(window).quantile(0.05)
    features['cvar_95'] = returns.rolling(window).apply(
        lambda x: x[x <= np.quantile(x, 0.05)].mean(), raw=True
    )
    
    # Volume-based (if available)
    if volume is not None:
        features['volume_momentum'] = volume_momentum(volume, window)
    
    return features.dropna()


if __name__ == "__main__":
    # Example: Compute observables for SPY
    import yfinance as yf
    
    print("Fetching SPY data...")
    data = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)
    
    # Compute observable map
    features = observable_map(
        prices=data['Adj Close'],
        volume=data['Volume'],
        window=60
    )
    
    print("\nObservable Map (last 5 days):")
    print(features.tail())
    
    print("\nFeature Statistics:")
    print(features.describe().round(4))
    
    # Compare regimes
    pre_covid = features['2020-01':'2020-02']
    covid_crash = features['2020-03':'2020-04']
    
    print("\n" + "="*60)
    print("Regime Comparison: Pre-COVID vs COVID Crash")
    print("="*60)
    print(f"\n{'Feature':<20} {'Pre-COVID':>15} {'COVID Crash':>15}")
    print("-"*60)
    for col in features.columns:
        pre = pre_covid[col].mean()
        crash = covid_crash[col].mean()
        print(f"{col:<20} {pre:>15.4f} {crash:>15.4f}")
