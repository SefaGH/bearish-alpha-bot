# Feature Analysis Tool

A comprehensive tool for ML feature quality assessment and selection in the Bearish Alpha Bot project.

## Overview

The Feature Analysis Tool (`scripts/analyze_features.py`) analyzes feature quality by examining variance and correlation with labels, then generates feature selection masks to improve ML model performance by removing low-quality features.

## Features

- **Variance Analysis**: Identifies features with low variance (< 0.01 by default)
- **Correlation Analysis**: Calculates Spearman correlation with labels to measure predictive power
- **Feature Selection**: Applies dual criteria (variance + correlation) to select high-quality features
- **Automated Reporting**: Generates detailed markdown reports with statistics and recommendations
- **Flexible Thresholds**: Customizable variance and correlation thresholds
- **Batch Processing**: Analyzes all features in a single run

## Installation

The tool requires Python 3.11 and the dependencies listed in `requirements.txt`:

```bash
# Ensure Python 3.11 is installed
python3.11 --version

# Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run full analysis with feature selection and report generation:

```bash
python scripts/analyze_features.py --analyze --select --report
```

### Command-Line Options

- `--analyze`: Run full analysis (variance + correlation)
- `--select`: Select features and save mask
- `--report`: Generate markdown report
- `--variance-threshold FLOAT`: Set variance threshold (default: 0.01)
- `--correlation-threshold FLOAT`: Set correlation threshold (default: 0.05)
- `--data-path PATH`: Path to training data NPZ file

### Examples

**Analyze features only:**
```bash
python scripts/analyze_features.py --analyze
```

**Select features with custom thresholds:**
```bash
python scripts/analyze_features.py --select --variance-threshold 0.1 --correlation-threshold 0.1
```

**Generate report from custom data:**
```bash
python scripts/analyze_features.py --analyze --report --data-path data/cache/custom_data.npz
```

## Data Format

The tool expects training data in NPZ format with the following structure:

```python
{
    'features': np.ndarray,  # Shape: (n_samples, n_features)
    'labels': np.ndarray,    # Shape: (n_samples,)
    'feature_names': list    # Optional: List of feature names
}
```

## Outputs

### 1. Feature Selection Mask
**Location:** `data/cache/feature_selection_mask.npy`

A boolean NumPy array indicating which features to keep:
```python
import numpy as np
mask = np.load('data/cache/feature_selection_mask.npy')
X_filtered = X[:, mask]  # Apply mask to features
```

### 2. Feature Metadata
**Location:** `data/cache/feature_selection_metadata.json`

JSON file containing:
- Number of samples and features
- Selection thresholds
- Selected and rejected feature lists
- Detailed statistics for each feature

### 3. Analysis Report
**Location:** `logs/feature_analysis_report.md`

Markdown report with:
- Variance statistics and distribution
- Correlation statistics and distribution
- Top 15 features by correlation
- Feature selection summary
- Actionable recommendations

## Selection Criteria

Features are selected based on **dual criteria**:

1. **Variance >= threshold** (default: 0.01)
2. **|Correlation| >= threshold** (default: 0.05)

Both conditions must be met for a feature to be selected.

## Example Results

Based on test data with 7,200 samples and 42 features:

```
Variance Analysis:
  Low-variance features (< 0.01): 22 (52.4%)
  
Correlation Analysis:
  Weak predictive features (|corr| < 0.05): 32 (76.2%)
  Strong predictive features (|corr| > 0.10): 10 (23.8%)
  
Feature Selection:
  ✅ Selected: 10/42 (23.8%)
  ❌ Rejected: 32/42 (76.2%)
  
Expected Benefits:
  - Reduced noise and overfitting
  - Faster training time
  - Better generalization
  - Estimated accuracy improvement: +10-15%
```

## Integration with Training Pipeline

Use the feature mask in your training pipeline:

```python
import numpy as np

# Load feature mask
mask = np.load('data/cache/feature_selection_mask.npy')

# Apply to training data
X_train_filtered = X_train[:, mask]
X_test_filtered = X_test[:, mask]

# Train model with filtered features
model.fit(X_train_filtered, y_train)
```

## Testing

Run the test suite:

```bash
pytest tests/test_feature_analyzer.py -v
```

All 14 tests should pass:
- Initialization tests
- Data loading tests
- Variance analysis tests
- Correlation analysis tests
- Feature selection tests
- Output generation tests
- Full workflow tests

## Implementation Details

### FeatureAnalyzer Class

**Methods:**
- `__init__(data_path, variance_threshold, correlation_threshold)`: Initialize analyzer
- `load_data()`: Load training data from NPZ file
- `analyze_variance()`: Calculate variance statistics
- `analyze_correlations()`: Calculate Spearman correlations
- `select_features()`: Apply selection criteria
- `save_feature_mask(output_dir)`: Save mask and metadata
- `generate_report(output_dir)`: Generate markdown report

### Key Technologies

- **NumPy**: Efficient numerical computations
- **SciPy**: Spearman correlation calculation
- **Python 3.11**: Modern Python features and performance

## Troubleshooting

### Data Not Found
```
ERROR - Data file not found: data/cache/BTC-USDT_training_data.npz
```
**Solution**: Ensure training data exists or specify correct path with `--data-path`

### Invalid Data Format
```
ERROR - Failed to load data: 'features' not in data
```
**Solution**: Verify NPZ file contains 'features' and 'labels' arrays

### No Features Selected
```
WARNING - No features meet selection criteria
```
**Solution**: Try lowering thresholds with `--variance-threshold` and `--correlation-threshold`

## Best Practices

1. **Run Before Training**: Always run feature analysis before training new models
2. **Review Reports**: Carefully review generated reports to understand feature characteristics
3. **Adjust Thresholds**: Experiment with different thresholds based on your data
4. **Track Results**: Monitor model performance improvements after feature selection
5. **Iterate**: Re-run analysis periodically as data evolves

## Contributing

To add new analysis features:

1. Add method to `FeatureAnalyzer` class
2. Update command-line interface in `main()`
3. Add corresponding unit tests
4. Update documentation

## License

This tool is part of the Bearish Alpha Bot project and follows the project's MIT license.

## Support

For issues or questions:
1. Check the generated report for insights
2. Review test cases for usage examples
3. Consult the project documentation
4. Open an issue on GitHub
