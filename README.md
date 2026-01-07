# Market Regime Detection via Measure-Theoretic Framework

Extension of physics research on stochastic process classification to financial time series. Applies measure-theoretic framework from "Universality Classes as Concentrating Measures in Observable Space" to market regime detection.

## Theoretical Foundation

**Physics Framework → Finance Application:**

| Physics Concept | Finance Translation |
|-----------------|---------------------|
| Surface height h(x,t) | Log price log(S_t) |
| Universality classes | Market regimes (bull, bear, crisis, sideways) |
| Observable map Φ | Technical indicators, volatility measures |
| Induced measure μ^Φ | Distribution of market states |
| Wasserstein distance W₁ | Regime distance metric |
| Anomaly detection | Regime change detection |

## Key Innovation

Rather than classifying regimes by threshold rules (e.g., "20% decline = bear market"), this framework:

1. **Characterizes regimes geometrically** as distinct probability measures in observable space
2. **Detects transitions** via Wasserstein distance to reference regimes
3. **Quantifies regime distance** continuously (not binary classification)
4. **Identifies novel regimes** via anomaly detection (unsupervised)

## Features Implemented

- ✅ Observable map construction (returns, volatility, skewness, autocorrelation)
- ✅ Induced measure computation via kernel density estimation
- ✅ Wasserstein-1 distance calculation
- ✅ Isolation Forest anomaly detection for regime changes
- ✅ Rolling regime detection with lookback windows
- ✅ Visualization of regime spaces

## Mathematical Framework

### Induced Measures

For price process S_t, define observable map:

```
Φ(S) = [σ(returns), skewness, kurtosis, autocorr, ...]
```

The induced measure μ^Φ_T captures the distribution of these observables over window [t-T, t].

### Regime Distance

Two market periods belong to same regime if:

```
W₁(μ^Φ₁, μ^Φ₂) < ε
```

where W₁ is Wasserstein-1 distance and ε is threshold.

### Concentration Property

**Conjecture (from physics paper):** As sample size T → ∞, measures concentrate:

```
δ(T) → 0  where δ is effective diameter
```

**Finance interpretation:** Longer time windows give more reliable regime classification.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from regime_detection import RegimeDetector
import yfinance as yf

# Fetch data
data = yf.download('SPY', start='2015-01-01', end='2024-01-01')

# Initialize detector
detector = RegimeDetector(
    window_size=252,  # 1 year
    observables=['volatility', 'skewness', 'autocorr']
)

# Train on normal period
detector.fit(data['2015':'2019'])

# Detect regime changes
changes = detector.detect_regime_changes(data['2020':'2024'])

# Compute regime distances
distances = detector.regime_distance(data['2020':'2024'])
```

## Modules

| Module | Description |
|--------|-------------|
| `observables.py` | Feature extraction (volatility, moments, correlations) |
| `measures.py` | Induced measure computation via KDE |
| `wasserstein.py` | W₁ distance calculation using POT library |
| `regime_detector.py` | Main anomaly detection pipeline |
| `visualization.py` | Regime space plots, distance timeseries |
| `backtesting.py` | Trading strategy based on regime signals |

## Theoretical Extensions

### From Physics Paper

**Proven for Edwards-Wilkinson (linear diffusion):**
- Concentration rate: δ(L) ~ L^(-1/2) via CLT

**Conjectured for KPZ (nonlinear growth):**
- Separation persists under generic projections (Johnson-Lindenstrauss)

**Financial analogue:**
- Linear market models (Brownian motion) → proven concentration
- Nonlinear/jump processes → conjectured but empirically testable

### Novel Contributions

1. **Observable selection:** Which indicators maximize regime separation?
2. **Optimal transport for finance:** Wasserstein distance as risk metric
3. **Unsupervised regime discovery:** No need to pre-define "bear market"
4. **Crossover quantification:** Measure "how bearish" continuously

## Empirical Results

**Physics validation:**
- 100% detection of unknown universality classes
- 2× SNR improvement vs traditional methods

**Finance application (preliminary):**
- Successfully detects 2020 COVID crash as anomaly
- Identifies 2022 bear market transition
- Lower false positive rate than volatility thresholds

## Connection to Existing Methods

| Method | Our Framework |
|--------|---------------|
| Hidden Markov Models | Geometric characterization of states |
| Regime-switching models | Continuous distance metric |
| Volatility thresholds | Multivariate observable space |
| Technical analysis | Measure-theoretic foundation |

## Academic References

**Primary theory paper:**
- Bentley, A. "Universality Classes as Concentrating Measures in Observable Space: A Geometric Framework for Non-Equilibrium Critical Phenomena" (2026, in preparation)

**Optimal transport in finance:**
- Peyré & Cuturi, "Computational Optimal Transport" (2019)
- Cont & Tankov, "Financial Modelling with Jump Processes" (2004)

**Anomaly detection:**
- Liu et al., "Isolation Forest" (2008)

## Future Work

- [ ] Extend to multivariate regime detection (equities + bonds + VIX)
- [ ] Incorporate transaction costs in backtesting
- [ ] Information-theoretic optimal observable selection
- [ ] Real-time streaming detection
- [ ] Connection to regime-switching GARCH models

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- POT (Python Optimal Transport)
- Scikit-learn
- yfinance, matplotlib

## License

MIT

---

**Note:** This project extends theoretical physics research to quantitative finance. The measure-theoretic framework is rigorously proven for certain stochastic systems (Edwards-Wilkinson) and conjectured for others (KPZ). Financial markets exhibit additional complexity (non-stationarity, regime-dependent volatility, fat tails) requiring empirical validation.
