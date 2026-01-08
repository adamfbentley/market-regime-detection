"""
Visualization tools for market regime detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def plot_wasserstein_timeline(distance_df, anomalies=True, title="Market Regime Transitions"):
    """
    Plot Wasserstein distance timeline with anomaly highlighting.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Wasserstein distance
    ax = axes[0]
    ax.plot(distance_df.index, distance_df['wasserstein'], 
            color='steelblue', linewidth=1.5, label='W₁ distance')
    
    if anomalies and 'anomaly' in distance_df.columns:
        anomaly_mask = distance_df['anomaly'].fillna(False).astype(bool)
        anomaly_dates = distance_df[anomaly_mask].index
        ax.scatter(anomaly_dates, 
                  distance_df.loc[anomaly_dates, 'wasserstein'],
                  color='red', s=50, zorder=5, label='Anomaly', alpha=0.7)
    
    ax.set_ylabel('Wasserstein Distance W₁', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Effective diameter
    ax = axes[1]
    ax.plot(distance_df.index, distance_df['diameter'], 
            color='darkorange', linewidth=1.5, label='Effective δ')
    
    if anomalies and 'anomaly' in distance_df.columns:
        ax.scatter(anomaly_dates,
                  distance_df.loc[anomaly_dates, 'diameter'],
                  color='red', s=50, zorder=5, alpha=0.7)
    
    ax.set_ylabel('Effective Diameter δ', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Anomaly score
    ax = axes[2]
    if 'anomaly_score' in distance_df.columns:
        ax.plot(distance_df.index, distance_df['anomaly_score'],
                color='darkgreen', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.fill_between(distance_df.index, distance_df['anomaly_score'], 0,
                        where=distance_df['anomaly_score'] < 0,
                        color='red', alpha=0.2, label='Anomalous')
    
    ax.set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_feature_space(features, anomalies=None, dims=(0, 1), title="Observable Space"):
    """
    2D projection of observable space with regime coloring.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    feature_cols = features.columns
    x_col = feature_cols[dims[0]]
    y_col = feature_cols[dims[1]]
    
    # Plot all points
    ax.scatter(features[x_col], features[y_col], 
              c='lightblue', s=20, alpha=0.5, label='Normal')
    
    # Highlight anomalies
    if anomalies is not None:
        anomaly_idx = features.index.isin(anomalies)
        ax.scatter(features.loc[anomaly_idx, x_col],
                  features.loc[anomaly_idx, y_col],
                  c='red', s=50, alpha=0.7, label='Anomalous', zorder=5)
    
    ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_pca(features, anomalies=None):
    """
    PCA projection of full observable space.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features.dropna())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normal points
    ax.scatter(X_pca[:, 0], X_pca[:, 1], 
              c='lightblue', s=20, alpha=0.5, label='Normal')
    
    # Anomalies
    if anomalies is not None:
        clean_features = features.dropna()
        anomaly_mask = clean_features.index.isin(anomalies)
        ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                  c='red', s=50, alpha=0.7, label='Anomalous', zorder=5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
                 fontsize=11, fontweight='bold')
    ax.set_title('PCA Projection of Observable Space', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_measure_concentration(features_dict, title="Measure Concentration Test"):
    """
    Compare effective diameters across different regimes/periods.
    
    features_dict : dict of {regime_name: features_df}
    """
    from measures import compute_induced_measure, effective_diameter
    
    regimes = []
    diameters = []
    
    for name, features in features_dict.items():
        mu = compute_induced_measure(features)
        delta = effective_diameter(mu)
        regimes.append(name)
        diameters.append(delta)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(regimes)))
    bars = ax.bar(regimes, diameters, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Effective Diameter δ', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, delta in zip(bars, diameters):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{delta:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_distance_matrix(features_dict):
    """
    Heatmap of pairwise Wasserstein distances between regimes.
    """
    from measures import compute_induced_measure, wasserstein_distance
    
    regimes = list(features_dict.keys())
    n = len(regimes)
    
    # Compute measures
    measures = {}
    for name, features in features_dict.items():
        measures[name] = compute_induced_measure(features)
    
    # Distance matrix
    W_matrix = np.zeros((n, n))
    for i, name_i in enumerate(regimes):
        for j, name_j in enumerate(regimes):
            if i <= j:
                W = wasserstein_distance(measures[name_i], measures[name_j], 
                                        p=1, n_samples=500)
                W_matrix[i, j] = W
                W_matrix[j, i] = W
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(W_matrix, cmap='YlOrRd', aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(regimes, rotation=45, ha='right')
    ax.set_yticklabels(regimes)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wasserstein Distance W₁', fontsize=11, fontweight='bold')
    
    # Annotate with values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{W_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Regime Distance Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_price_with_regimes(prices, distance_df, title="Price with Regime Overlay"):
    """
    Price chart with anomaly/regime highlighting.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Price
    ax = axes[0]
    ax.plot(prices.index, prices, color='black', linewidth=1.5, label='Price')
    
    if 'anomaly' in distance_df.columns:
        anomaly_mask = distance_df['anomaly'].fillna(False).astype(bool)
        anomaly_dates = distance_df[anomaly_mask].index
        # Highlight anomalous periods
        for date in anomaly_dates:
            ax.axvspan(date, date + pd.Timedelta(days=1), 
                      color='red', alpha=0.1)
    
    ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Wasserstein distance
    ax = axes[1]
    ax.plot(distance_df.index, distance_df['wasserstein'], 
            color='steelblue', linewidth=1.5, label='W₁ distance')
    
    if 'anomaly' in distance_df.columns:
        ax.scatter(anomaly_dates,
                  distance_df.loc[anomaly_dates, 'wasserstein'],
                  color='red', s=50, zorder=5, alpha=0.7)
    
    ax.set_ylabel('Wasserstein Distance', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import yfinance as yf
    import os
    from observables import observable_map
    from regime_detector import RegimeDetector
    
    os.makedirs('plots', exist_ok=True)
    
    print("="*60)
    print("MARKET REGIME DETECTION - GENERATING PLOTS")
    print("="*60)
    
    # Data
    print("\nFetching SPY data...")
    spy = yf.download('SPY', start='2019-01-01', end='2024-01-01', progress=False, auto_adjust=True)
    
    # Train detector
    print("Training regime detector...")
    detector = RegimeDetector(window=60, reference_period=252)
    detector.fit(spy['Close'], spy['Volume'], end_date='2021-12-31')
    
    # Generate plots
    print("\n1. Wasserstein Distance Timeline...")
    fig1 = plot_wasserstein_timeline(
        detector.distance_df, 
        title="SPY Regime Detection: Wasserstein Distance from Reference (2019-2021)"
    )
    plt.savefig('plots/regime_timeline.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved plots/regime_timeline.png")
    
    print("2. Price with Regime Overlay...")
    fig2 = plot_price_with_regimes(
        spy['Close'].loc[:'2021-12-31'],
        detector.distance_df,
        title="SPY Price with Detected Regime Anomalies"
    )
    plt.savefig('plots/price_regimes.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved plots/price_regimes.png")
    
    # Feature space
    print("3. Observable Feature Space...")
    features = observable_map(spy['Close'], spy['Volume'], window=60)
    anomaly_mask = detector.distance_df['anomaly'].fillna(False).astype(bool)
    anomalies = detector.distance_df[anomaly_mask].index
    
    fig3 = plot_feature_space(
        features.loc[:'2021-12-31'],
        anomalies=anomalies,
        dims=(0, 1),
        title="Observable Space: Normal vs Anomalous Regimes"
    )
    plt.savefig('plots/feature_space.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved plots/feature_space.png")
    
    # PCA
    print("4. PCA Projection...")
    fig4 = plot_regime_pca(features.loc[:'2021-12-31'], anomalies=anomalies)
    plt.savefig('plots/pca_projection.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved plots/pca_projection.png")
    
    print("\n" + "="*60)
    print("✓ All market-regime-detection plots generated!")
    print("="*60)
    plt.close('all')
