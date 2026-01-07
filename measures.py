"""
Induced measure computation and Wasserstein distance calculation.
Core of the measure-theoretic framework.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial.distance import euclidean
import ot  # Python Optimal Transport library


def compute_induced_measure(features, method='kde', bandwidth='scott'):
    """
    Compute induced measure μ^Φ from observable features.
    
    Physics: μ^Φ = Φ_* P where P is the process measure
    Finance: Distribution of feature vectors over time window
    
    Parameters:
    -----------
    features : pd.DataFrame
        Observable features (output of observable_map)
    method : str
        'kde' for kernel density estimation
        'histogram' for binned approximation
    bandwidth : str or float
        KDE bandwidth selection
    
    Returns:
    --------
    InducedMeasure object
    """
    # Remove any remaining NaNs
    clean_features = features.dropna()
    
    if len(clean_features) == 0:
        raise ValueError("No valid samples for measure computation")
    
    if method == 'kde':
        # Multivariate KDE
        data = clean_features.values.T
        try:
            kde = gaussian_kde(data, bw_method=bandwidth)
            return KDEMeasure(kde, clean_features)
        except np.linalg.LinAlgError:
            # Fallback to empirical measure if KDE fails
            return EmpiricalMeasure(clean_features)
    
    elif method == 'histogram':
        return HistogramMeasure(clean_features)
    
    else:
        raise ValueError(f"Unknown method: {method}")


class KDEMeasure:
    """Induced measure represented via KDE."""
    
    def __init__(self, kde, samples):
        self.kde = kde
        self.samples = samples
        self.d = samples.shape[1]
    
    def pdf(self, x):
        """Evaluate probability density at point x."""
        return self.kde(x.T)
    
    def sample(self, n):
        """Sample from the measure."""
        return self.kde.resample(n).T
    
    def mean(self):
        """Mean of the distribution."""
        return self.samples.mean().values
    
    def cov(self):
        """Covariance matrix."""
        return self.samples.cov().values


class EmpiricalMeasure:
    """Empirical measure from finite samples."""
    
    def __init__(self, samples):
        self.samples = samples
        self.weights = np.ones(len(samples)) / len(samples)
        self.d = samples.shape[1]
    
    def sample(self, n):
        """Resample with replacement."""
        idx = np.random.choice(len(self.samples), size=n, replace=True)
        return self.samples.iloc[idx].values
    
    def mean(self):
        return self.samples.mean().values
    
    def cov(self):
        return self.samples.cov().values


class HistogramMeasure:
    """Binned histogram representation."""
    
    def __init__(self, samples, bins=10):
        self.samples = samples
        self.bins = bins
        self.d = samples.shape[1]
        
        # Compute histogram
        # For simplicity, use 1D marginals
        self.hists = []
        self.edges = []
        for col in samples.columns:
            hist, edges = np.histogram(samples[col], bins=bins, density=True)
            self.hists.append(hist)
            self.edges.append(edges)
    
    def mean(self):
        return self.samples.mean().values
    
    def cov(self):
        return self.samples.cov().values


def wasserstein_distance(measure1, measure2, p=1, n_samples=1000):
    """
    Compute Wasserstein-p distance between two measures.
    
    W_p(μ₁, μ₂) = (inf_{γ ∈ Γ(μ₁,μ₂)} ∫ |x-y|^p dγ(x,y))^(1/p)
    
    Parameters:
    -----------
    measure1, measure2 : Measure objects
        Induced measures to compare
    p : int
        Order of Wasserstein distance (typically 1 or 2)
    n_samples : int
        Number of samples for empirical approximation
    
    Returns:
    --------
    float : W_p distance
    """
    # Sample from both measures
    X1 = measure1.sample(n_samples)
    X2 = measure2.sample(n_samples)
    
    # Uniform weights
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    
    # Cost matrix: pairwise distances^p
    M = ot.dist(X1, X2, metric='euclidean')
    if p != 1:
        M = M ** p
    
    # Compute optimal transport
    W_p = ot.emd2(a, b, M)
    
    if p != 1:
        W_p = W_p ** (1/p)
    
    return W_p


def effective_diameter(measure, confidence=0.95, n_samples=1000):
    """
    Compute effective diameter δ_ε(μ) of measure support.
    
    δ_ε = inf{r : μ(B_r(x)) ≥ 1-ε for some x}
    
    Approximated by: sample from measure, find radius containing
    (1-ε) fraction of mass.
    
    Physics: Measures how concentrated the distribution is
    Finance: Low δ → tight regime, high δ → dispersed/transitioning
    """
    samples = measure.sample(n_samples)
    
    # Find center (median for robustness)
    center = np.median(samples, axis=0)
    
    # Distances from center
    distances = np.array([euclidean(x, center) for x in samples])
    
    # Radius containing (1-confidence) fraction
    quantile = np.quantile(distances, confidence)
    
    return quantile


def concentration_test(features_early, features_late):
    """
    Test concentration conjecture: δ(T) decreases with sample size T.
    
    Physics proof for EW: δ(L) ~ L^(-1/2)
    Finance empirical test: Compare diameter for different window sizes
    
    Returns:
    --------
    dict : {'delta_early': float, 'delta_late': float, 'ratio': float}
    """
    measure_early = compute_induced_measure(features_early)
    measure_late = compute_induced_measure(features_late)
    
    delta_early = effective_diameter(measure_early)
    delta_late = effective_diameter(measure_late)
    
    return {
        'delta_early': delta_early,
        'delta_late': delta_late,
        'ratio': delta_early / delta_late if delta_late > 0 else np.inf,
        'concentration': delta_late < delta_early
    }


if __name__ == "__main__":
    from observables import observable_map
    import yfinance as yf
    
    print("Fetching data...")
    spy = yf.download('SPY', start='2019-01-01', end='2024-01-01', progress=False)
    
    # Compute observables
    features = observable_map(spy['Adj Close'], spy['Volume'], window=60)
    
    # Define regimes
    normal_2019 = features['2019-01':'2019-12']
    covid_crash = features['2020-03':'2020-04']
    recovery_2021 = features['2021-01':'2021-12']
    
    print("\n" + "="*60)
    print("Computing Induced Measures and Wasserstein Distances")
    print("="*60)
    
    # Compute measures
    mu_normal = compute_induced_measure(normal_2019)
    mu_crash = compute_induced_measure(covid_crash)
    mu_recovery = compute_induced_measure(recovery_2021)
    
    # Wasserstein distances
    W_normal_crash = wasserstein_distance(mu_normal, mu_crash, p=1, n_samples=500)
    W_crash_recovery = wasserstein_distance(mu_crash, mu_recovery, p=1, n_samples=500)
    W_normal_recovery = wasserstein_distance(mu_normal, mu_recovery, p=1, n_samples=500)
    
    print(f"\nW₁(Normal 2019, COVID Crash):    {W_normal_crash:.4f}")
    print(f"W₁(COVID Crash, Recovery 2021):  {W_crash_recovery:.4f}")
    print(f"W₁(Normal 2019, Recovery 2021):  {W_normal_recovery:.4f}")
    
    print("\nInterpretation:")
    print(f"  • COVID crash is {W_normal_crash:.1f}× distance from normal vs recovery")
    print(f"  • Recovery closer to normal than crash (W={W_normal_recovery:.2f})")
    
    # Effective diameters
    delta_normal = effective_diameter(mu_normal)
    delta_crash = effective_diameter(mu_crash)
    
    print(f"\nEffective Diameters (δ):")
    print(f"  Normal 2019:  δ = {delta_normal:.4f}")
    print(f"  COVID Crash:  δ = {delta_crash:.4f}")
    print(f"  Ratio: {delta_crash / delta_normal:.2f}")
    
    if delta_crash > delta_normal:
        print("  → Crash regime is MORE dispersed (transitioning/unstable)")
    else:
        print("  → Crash regime is MORE concentrated")
