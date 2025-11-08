"""
Create test training data for feature analysis validation.

This script generates synthetic training data with known properties
for testing the feature analyzer.
"""

import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("data/cache")
output_dir.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 7200
n_features = 42

# Generate features with different characteristics
features = np.zeros((n_samples, n_features))

# Features 0-9: High variance, high correlation (good features)
for i in range(10):
    features[:, i] = np.random.randn(n_samples) * 5 + np.linspace(-10, 10, n_samples)

# Features 10-19: High variance, low correlation (noisy features)
for i in range(10, 20):
    features[:, i] = np.random.randn(n_samples) * 5

# Features 20-30: Low variance, high correlation (constant-ish features)
for i in range(20, 31):
    features[:, i] = np.random.randn(n_samples) * 0.005 + 1.0

# Features 31-41: Low variance, low correlation (useless features)
for i in range(31, 42):
    features[:, i] = np.random.randn(n_samples) * 0.001

# Generate labels correlated with good features
labels = np.zeros(n_samples)
for i in range(10):  # Use first 10 features
    labels += features[:, i] * (0.1 + i * 0.01)

# Add some noise to labels
labels += np.random.randn(n_samples) * 2

# Normalize labels to [0, 1, 2] (simulating regime labels)
labels = np.digitize(labels, bins=[np.percentile(labels, 33), np.percentile(labels, 67)])

# Generate feature names
feature_names = [f"feature_{i}" for i in range(n_features)]

# Save to NPZ file
output_file = output_dir / "BTC-USDT_training_data.npz"
np.savez(
    output_file,
    features=features,
    labels=labels,
    feature_names=feature_names
)

print(f"✅ Created test training data:")
print(f"   File: {output_file}")
print(f"   Samples: {n_samples}")
print(f"   Features: {n_features}")
print(f"   Labels: {len(np.unique(labels))} classes")
print(f"\nFeature characteristics:")
print(f"   Features 0-9:   High variance, high correlation (GOOD)")
print(f"   Features 10-19: High variance, low correlation (NOISY)")
print(f"   Features 20-30: Low variance, high correlation (CONSTANT)")
print(f"   Features 31-41: Low variance, low correlation (USELESS)")
