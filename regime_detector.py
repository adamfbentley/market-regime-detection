"""
Market Regime Detection using Measure-Theoretic Framework.

Main pipeline: Observable Map → Induced Measures → Wasserstein Distances → Anomaly Detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from observables import observable_map
from measures import compute_induced_measure, wasserstein_distance, effective_diameter


class RegimeDetector:
    """
    Detect market regime transitions using Wasserstein distances.
    
    Physics Framework:
    ------------------
    - Observable map Φ: price trajectories → feature space
    - Induced measures μ^Φ concentrated around regime centroids
    - W₁ distance separates regimes
    
    Financial Application:
    ----------------------
    - Normal regimes: concentrated measures (small δ)
    - Transitions/crises: dispersed measures (large δ)
    - Isolation Forest flags anomalous distances
    """
    
    def __init__(self, 
                 window=60,
                 reference_period=252,
                 contamination=0.1,
                 n_estimators=100):
        """
        Parameters:
        -----------
        window : int
            Rolling window for observable computation (days)
        reference_period : int
            Lookback for reference measure (days)
        contamination : float
            Expected fraction of anomalies (Isolation Forest)
        n_estimators : int
            Number of trees in Isolation Forest
        """
        self.window = window
        self.reference_period = reference_period
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        
        self.reference_measure = None
        self.features_history = []
        self.distances_history = []
    
    def fit(self, prices, volumes, start_date=None, end_date=None):
        """
        Fit detector on training period.
        
        1. Compute observable map
        2. Establish reference measure (μ_ref)
        3. Compute rolling Wasserstein distances
        4. Train Isolation Forest on distance statistics
        """
        # Slice training period
        if start_date or end_date:
            prices = prices.loc[start_date:end_date]
            volumes = volumes.loc[start_date:end_date]
        
        print(f"Computing observables (window={self.window})...")
        features = observable_map(prices, volumes, window=self.window)
        
        # Define reference measure from first reference_period days
        print(f"Establishing reference measure ({self.reference_period} days)...")
        reference_features = features.iloc[:self.reference_period]
        self.reference_measure = compute_induced_measure(reference_features)
        self.reference_mean = reference_features.mean().values
        self.reference_cov = reference_features.cov().values
        
        # Compute rolling Wasserstein distances
        print("Computing Wasserstein distances...")
        distances = []
        diameters = []
        
        for i in range(len(features) - self.window + 1):
            window_features = features.iloc[i:i+self.window]
            
            if len(window_features.dropna()) < 10:  # Require minimum samples
                continue
            
            # Induced measure for this window
            mu_window = compute_induced_measure(window_features)
            
            # W₁ distance from reference
            try:
                W = wasserstein_distance(self.reference_measure, mu_window, p=1, n_samples=500)
                delta = effective_diameter(mu_window)
                
                distances.append({
                    'date': features.index[i + self.window - 1],
                    'wasserstein': W,
                    'diameter': delta
                })
                
            except Exception as e:
                print(f"Warning: distance computation failed at {features.index[i]}: {e}")
                continue
        
        self.distance_df = pd.DataFrame(distances).set_index('date')
        
        # Augment with statistical features
        self.distance_df['W_rolling_mean'] = self.distance_df['wasserstein'].rolling(20).mean()
        self.distance_df['W_rolling_std'] = self.distance_df['wasserstein'].rolling(20).std()
        self.distance_df['W_z_score'] = (
            (self.distance_df['wasserstein'] - self.distance_df['W_rolling_mean']) 
            / self.distance_df['W_rolling_std']
        )
        
        # Train Isolation Forest
        print("Training anomaly detector...")
        X_train = self.distance_df[['wasserstein', 'diameter', 'W_z_score']].dropna()
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.detector.fit(X_scaled)
        
        # Decision scores
        scores = self.detector.decision_function(X_scaled)
        self.distance_df.loc[X_train.index, 'anomaly_score'] = scores
        self.distance_df.loc[X_train.index, 'anomaly'] = self.detector.predict(X_scaled) == -1
        
        print(f"✓ Training complete. {self.distance_df['anomaly'].sum()} anomalies detected.")
        
        return self
    
    def predict(self, prices, volumes):
        """
        Predict regime anomalies for new data.
        
        Returns:
        --------
        pd.DataFrame with columns:
          - wasserstein: W₁ distance from reference
          - diameter: effective diameter δ
          - anomaly: boolean flag
          - anomaly_score: decision function output
        """
        if self.reference_measure is None:
            raise ValueError("Must call fit() before predict()")
        
        features = observable_map(prices, volumes, window=self.window)
        
        distances = []
        for i in range(len(features) - self.window + 1):
            window_features = features.iloc[i:i+self.window]
            
            if len(window_features.dropna()) < 10:
                continue
            
            mu_window = compute_induced_measure(window_features)
            
            try:
                W = wasserstein_distance(self.reference_measure, mu_window, p=1, n_samples=500)
                delta = effective_diameter(mu_window)
                
                distances.append({
                    'date': features.index[i + self.window - 1],
                    'wasserstein': W,
                    'diameter': delta
                })
            except:
                continue
        
        df = pd.DataFrame(distances).set_index('date')
        
        # Compute rolling statistics
        df['W_rolling_mean'] = df['wasserstein'].rolling(20).mean()
        df['W_rolling_std'] = df['wasserstein'].rolling(20).std()
        df['W_z_score'] = (df['wasserstein'] - df['W_rolling_mean']) / df['W_rolling_std']
        
        # Predict anomalies
        X = df[['wasserstein', 'diameter', 'W_z_score']].dropna()
        X_scaled = self.scaler.transform(X)
        
        df.loc[X.index, 'anomaly_score'] = self.detector.decision_function(X_scaled)
        df.loc[X.index, 'anomaly'] = self.detector.predict(X_scaled) == -1
        
        return df
    
    def get_regime_summary(self):
        """Summary statistics of detected regimes."""
        if self.distance_df is None:
            raise ValueError("Must call fit() first")
        
        df = self.distance_df.dropna()
        
        normal = df[df['anomaly'] == False]
        anomalous = df[df['anomaly'] == True]
        
        summary = {
            'n_samples': len(df),
            'n_anomalies': len(anomalous),
            'anomaly_rate': len(anomalous) / len(df),
            'mean_W_normal': normal['wasserstein'].mean(),
            'mean_W_anomalous': anomalous['wasserstein'].mean(),
            'mean_delta_normal': normal['diameter'].mean(),
            'mean_delta_anomalous': anomalous['diameter'].mean(),
            'separation_ratio': (
                anomalous['wasserstein'].mean() / normal['wasserstein'].mean()
                if len(normal) > 0 and len(anomalous) > 0 else np.nan
            )
        }
        
        return summary


if __name__ == "__main__":
    import yfinance as yf
    
    print("="*80)
    print("MARKET REGIME DETECTION: Measure-Theoretic Framework")
    print("="*80)
    
    # Download data
    print("\nDownloading SPY data (2018-2024)...")
    spy = yf.download('SPY', start='2018-01-01', end='2024-01-01', progress=False)
    
    # Split train/test
    train_end = '2021-12-31'
    test_start = '2022-01-01'
    
    # Initialize detector
    detector = RegimeDetector(
        window=60,
        reference_period=252,  # 1 year reference
        contamination=0.1
    )
    
    # Train on pre-COVID and recovery period
    print(f"\n{'='*80}")
    print("TRAINING PHASE (2018-01-01 to 2021-12-31)")
    print(f"{'='*80}")
    detector.fit(
        spy['Adj Close'], 
        spy['Volume'],
        start_date='2018-01-01',
        end_date=train_end
    )
    
    # Get training summary
    summary = detector.get_regime_summary()
    print(f"\nTraining Summary:")
    print(f"  Total samples: {summary['n_samples']}")
    print(f"  Anomalies detected: {summary['n_anomalies']} ({summary['anomaly_rate']:.1%})")
    print(f"  Mean W₁ (normal): {summary['mean_W_normal']:.4f}")
    print(f"  Mean W₁ (anomalous): {summary['mean_W_anomalous']:.4f}")
    print(f"  Separation ratio: {summary['separation_ratio']:.2f}×")
    
    # Test on 2022-2023 (Fed hiking, bear market)
    print(f"\n{'='*80}")
    print("TESTING PHASE (2022-01-01 to 2024-01-01)")
    print(f"{'='*80}")
    predictions = detector.predict(
        spy['Adj Close'].loc[test_start:],
        spy['Volume'].loc[test_start:]
    )
    
    test_summary = {
        'n_samples': len(predictions.dropna()),
        'n_anomalies': predictions['anomaly'].sum(),
        'anomaly_rate': predictions['anomaly'].mean(),
        'mean_W': predictions['wasserstein'].mean()
    }
    
    print(f"\nTest Summary:")
    print(f"  Total samples: {test_summary['n_samples']}")
    print(f"  Anomalies: {test_summary['n_anomalies']} ({test_summary['anomaly_rate']:.1%})")
    print(f"  Mean W₁: {test_summary['mean_W']:.4f}")
    
    # Show anomalous periods
    anomalies = predictions[predictions['anomaly']].index
    if len(anomalies) > 0:
        print(f"\nAnomaly Periods Detected:")
        for date in anomalies[:10]:  # Show first 10
            W = predictions.loc[date, 'wasserstein']
            print(f"  {date.date()}: W₁ = {W:.4f}")
    
    print(f"\n{'='*80}")
    print("✓ Regime detection complete. Results in detector.distance_df")
    print(f"{'='*80}")
