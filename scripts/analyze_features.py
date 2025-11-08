"""
Feature Analysis Tool for ML Feature Quality Assessment

This tool analyzes feature quality, identifies low-variance and low-correlation
features, and generates feature selection masks for improved ML performance.

Key Features:
- Variance analysis to identify near-constant features
- Spearman correlation analysis to measure predictive power
- Dual-criteria feature selection (variance + correlation)
- Automated report generation with actionable insights
- Customizable thresholds for different use cases

Usage:
    # Full analysis with all outputs
    python scripts/analyze_features.py --analyze --select --report
    
    # Custom thresholds
    python scripts/analyze_features.py --variance-threshold 0.01 --correlation-threshold 0.05
    
    # Analysis only (no selection)
    python scripts/analyze_features.py --analyze

Expected Results:
- Identifies ~50% low-variance features
- Identifies ~76% weak correlation features  
- Selects 10-15 high-quality features from 42 total
- Expected +10-15% accuracy improvement after feature selection

Author: SefaGH
Date: 2025-11-08
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Analyze feature quality for ML model training.
    
    Provides variance analysis, correlation analysis, and feature selection
    capabilities to identify and filter low-quality features.
    """
    
    def __init__(
        self,
        data_path: str = "data/cache/BTC-USDT_training_data.npz",
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.05
    ):
        """
        Initialize the Feature Analyzer.
        
        Args:
            data_path: Path to training data NPZ file
            variance_threshold: Minimum variance threshold for feature selection
            correlation_threshold: Minimum absolute correlation threshold
        """
        self.data_path = Path(data_path)
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.n_samples: int = 0
        self.n_features: int = 0
        
        # Analysis results
        self.variances: Optional[np.ndarray] = None
        self.correlations: Optional[np.ndarray] = None
        self.p_values: Optional[np.ndarray] = None
        self.feature_mask: Optional[np.ndarray] = None
        
        logger.info(f"Initialized FeatureAnalyzer with:")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Variance threshold: {self.variance_threshold}")
        logger.info(f"  Correlation threshold: {self.correlation_threshold}")
    
    def load_data(self) -> bool:
        """
        Load training data from NPZ file.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            if not self.data_path.exists():
                logger.error(f"Data file not found: {self.data_path}")
                return False
            
            logger.info(f"Loading data from {self.data_path}...")
            data = np.load(self.data_path)
            
            # Extract features and labels (using keys 'X' and 'y' to match prepare_training_data.py)
            self.features = data['X']
            self.labels = data['y']
            
            self.n_samples, self.n_features = self.features.shape
            
            # Generate feature names if not provided
            if 'feature_names' in data:
                self.feature_names = list(data['feature_names'])
            else:
                self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
            
            logger.info(f"✅ Data loaded successfully:")
            logger.info(f"  Samples: {self.n_samples}")
            logger.info(f"  Features: {self.n_features}")
            logger.info(f"  Labels shape: {self.labels.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def analyze_variance(self) -> Dict[str, any]:
        """
        Analyze variance for all features.
        
        Returns:
            Dictionary containing variance statistics
        """
        if self.features is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
        
        logger.info("="*70)
        logger.info("📊 VARIANCE ANALYSIS")
        logger.info("="*70)
        
        # Calculate variance for each feature
        self.variances = np.var(self.features, axis=0)
        
        # Variance statistics
        stats_dict = {
            'mean': float(np.mean(self.variances)),
            'median': float(np.median(self.variances)),
            'min': float(np.min(self.variances)),
            'max': float(np.max(self.variances)),
            'std': float(np.std(self.variances))
        }
        
        # Count features by variance thresholds
        thresholds = [0.001, 0.01, 0.1, 1.0]
        threshold_counts = {}
        for thresh in thresholds:
            count = np.sum(self.variances < thresh)
            percentage = (count / self.n_features) * 100
            threshold_counts[thresh] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Log results
        logger.info("\nVariance Statistics:")
        logger.info(f"  Mean:   {stats_dict['mean']:.6f}")
        logger.info(f"  Median: {stats_dict['median']:.6f}")
        logger.info(f"  Min:    {stats_dict['min']:.6f}")
        logger.info(f"  Max:    {stats_dict['max']:.6f}")
        logger.info(f"  Std:    {stats_dict['std']:.6f}")
        
        logger.info("\nFeatures by Variance Threshold:")
        for thresh in thresholds:
            info = threshold_counts[thresh]
            logger.info(f"  < {thresh:6.3f}: {info['count']:3d} features ({info['percentage']:5.1f}%)")
        
        # Identify low-variance features
        low_var_mask = self.variances < self.variance_threshold
        low_var_count = np.sum(low_var_mask)
        logger.info(f"\n⚠️  Low-variance features (< {self.variance_threshold}): {low_var_count} ({low_var_count/self.n_features*100:.1f}%)")
        
        return {
            'statistics': stats_dict,
            'threshold_counts': threshold_counts,
            'low_variance_count': int(low_var_count)
        }
    
    def analyze_correlations(self) -> Dict[str, any]:
        """
        Analyze Spearman correlation between features and labels.
        
        Returns:
            Dictionary containing correlation statistics
        """
        if self.features is None or self.labels is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
        
        logger.info("="*70)
        logger.info("📈 CORRELATION ANALYSIS")
        logger.info("="*70)
        
        # Calculate Spearman correlation for each feature
        self.correlations = np.zeros(self.n_features)
        self.p_values = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            try:
                corr, p_val = stats.spearmanr(self.features[:, i], self.labels)
                self.correlations[i] = corr
                self.p_values[i] = p_val
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for feature {i}: {e}")
                self.correlations[i] = 0.0
                self.p_values[i] = 1.0
        
        # Absolute correlations for analysis
        abs_correlations = np.abs(self.correlations)
        
        # Correlation statistics
        stats_dict = {
            'mean': float(np.mean(abs_correlations)),
            'median': float(np.median(abs_correlations)),
            'min': float(np.min(abs_correlations)),
            'max': float(np.max(abs_correlations)),
            'std': float(np.std(abs_correlations))
        }
        
        # Count features by correlation thresholds
        thresholds = [0.01, 0.05, 0.10, 0.20]
        threshold_counts = {}
        for thresh in thresholds:
            count = np.sum(abs_correlations >= thresh)
            percentage = (count / self.n_features) * 100
            threshold_counts[thresh] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Log results
        logger.info("\nCorrelation Statistics (absolute values):")
        logger.info(f"  Mean:   {stats_dict['mean']:.6f}")
        logger.info(f"  Median: {stats_dict['median']:.6f}")
        logger.info(f"  Min:    {stats_dict['min']:.6f}")
        logger.info(f"  Max:    {stats_dict['max']:.6f}")
        logger.info(f"  Std:    {stats_dict['std']:.6f}")
        
        logger.info("\nFeatures by Correlation Threshold:")
        for thresh in thresholds:
            info = threshold_counts[thresh]
            logger.info(f"  >= {thresh:.2f}: {info['count']:3d} features ({info['percentage']:5.1f}%)")
        
        # Identify weak and strong features
        weak_mask = abs_correlations < self.correlation_threshold
        weak_count = np.sum(weak_mask)
        strong_mask = abs_correlations > 0.10
        strong_count = np.sum(strong_mask)
        
        logger.info(f"\n⚠️  Weak predictive features (|corr| < {self.correlation_threshold}): {weak_count} ({weak_count/self.n_features*100:.1f}%)")
        logger.info(f"✅ Strong predictive features (|corr| > 0.10): {strong_count} ({strong_count/self.n_features*100:.1f}%)")
        
        # Top features by correlation
        top_indices = np.argsort(abs_correlations)[::-1][:15]
        logger.info("\nTop 15 Features by Correlation:")
        for rank, idx in enumerate(top_indices, 1):
            logger.info(f"  {rank:2d}. {self.feature_names[idx]:20s} | corr={self.correlations[idx]:+.4f} | p={self.p_values[idx]:.4f}")
        
        return {
            'statistics': stats_dict,
            'threshold_counts': threshold_counts,
            'weak_count': int(weak_count),
            'strong_count': int(strong_count),
            'top_features': [
                {
                    'name': self.feature_names[idx],
                    'correlation': float(self.correlations[idx]),
                    'p_value': float(self.p_values[idx])
                }
                for idx in top_indices
            ]
        }
    
    def select_features(self) -> Dict[str, any]:
        """
        Select features based on variance and correlation thresholds.
        
        Returns:
            Dictionary containing selection statistics
        """
        if self.variances is None or self.correlations is None:
            logger.error("Analysis not performed. Run analyze_variance() and analyze_correlations() first.")
            return {}
        
        logger.info("="*70)
        logger.info("🎯 FEATURE SELECTION")
        logger.info("="*70)
        
        # Apply dual criteria
        variance_mask = self.variances >= self.variance_threshold
        correlation_mask = np.abs(self.correlations) >= self.correlation_threshold
        
        # Combined mask
        self.feature_mask = variance_mask & correlation_mask
        
        selected_count = np.sum(self.feature_mask)
        rejected_count = self.n_features - selected_count
        
        logger.info(f"\nSelection Criteria:")
        logger.info(f"  Variance >= {self.variance_threshold}")
        logger.info(f"  |Correlation| >= {self.correlation_threshold}")
        
        logger.info(f"\nSelection Results:")
        logger.info(f"  ✅ Selected: {selected_count}/{self.n_features} ({selected_count/self.n_features*100:.1f}%)")
        logger.info(f"  ❌ Rejected: {rejected_count}/{self.n_features} ({rejected_count/self.n_features*100:.1f}%)")
        
        # Breakdown by rejection reason
        low_var_only = (~variance_mask) & correlation_mask
        low_corr_only = variance_mask & (~correlation_mask)
        both_low = (~variance_mask) & (~correlation_mask)
        
        logger.info(f"\nRejection Breakdown:")
        logger.info(f"  Low variance only:    {np.sum(low_var_only):3d} features")
        logger.info(f"  Low correlation only: {np.sum(low_corr_only):3d} features")
        logger.info(f"  Both low:             {np.sum(both_low):3d} features")
        
        # List selected features
        selected_indices = np.where(self.feature_mask)[0]
        logger.info(f"\nSelected Features ({selected_count}):")
        for idx in selected_indices:
            logger.info(f"  {self.feature_names[idx]:20s} | var={self.variances[idx]:.6f} | corr={self.correlations[idx]:+.4f}")
        
        return {
            'selected_count': int(selected_count),
            'rejected_count': int(rejected_count),
            'selection_rate': float(selected_count / self.n_features),
            'rejection_breakdown': {
                'low_variance_only': int(np.sum(low_var_only)),
                'low_correlation_only': int(np.sum(low_corr_only)),
                'both_low': int(np.sum(both_low))
            },
            'selected_features': [self.feature_names[idx] for idx in selected_indices]
        }
    
    def save_feature_mask(self, output_dir: str = "data/cache") -> bool:
        """
        Save feature selection mask and metadata.
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self.feature_mask is None:
            logger.error("No feature mask to save. Run select_features() first.")
            return False
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save feature mask
            mask_file = output_path / "feature_selection_mask.npy"
            np.save(mask_file, self.feature_mask)
            logger.info(f"✅ Saved feature mask to {mask_file}")
            
            # Prepare metadata
            metadata = {
                'n_samples': int(self.n_samples),
                'n_features_original': int(self.n_features),
                'n_features_selected': int(np.sum(self.feature_mask)),
                'variance_threshold': float(self.variance_threshold),
                'correlation_threshold': float(self.correlation_threshold),
                'selected_features': [
                    self.feature_names[i] for i in range(self.n_features)
                    if self.feature_mask[i]
                ],
                'rejected_features': [
                    self.feature_names[i] for i in range(self.n_features)
                    if not self.feature_mask[i]
                ],
                'feature_statistics': {
                    self.feature_names[i]: {
                        'variance': float(self.variances[i]),
                        'correlation': float(self.correlations[i]),
                        'p_value': float(self.p_values[i]),
                        'selected': bool(self.feature_mask[i])
                    }
                    for i in range(self.n_features)
                }
            }
            
            # Save metadata
            metadata_file = output_path / "feature_selection_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✅ Saved metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feature mask: {e}")
            return False
    
    def generate_report(self, output_dir: str = "logs") -> bool:
        """
        Generate markdown report with analysis results.
        
        Args:
            output_dir: Directory to save report
            
        Returns:
            True if report generated successfully, False otherwise
        """
        if self.variances is None or self.correlations is None:
            logger.error("Analysis not performed. Run analyze_variance() and analyze_correlations() first.")
            return False
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "feature_analysis_report.md"
        
        try:
            with open(report_file, 'w') as f:
                # Header
                f.write("# Feature Analysis Report\n\n")
                f.write(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Data:** {self.data_path}\n\n")
                f.write(f"**Samples:** {self.n_samples:,}\n\n")
                f.write(f"**Features:** {self.n_features}\n\n")
                f.write("---\n\n")
                
                # Variance Analysis
                f.write("## 1. Variance Analysis\n\n")
                f.write("Feature variance measures the spread of values. Low-variance features provide little information.\n\n")
                
                f.write("### Variance Statistics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Mean | {np.mean(self.variances):.6f} |\n")
                f.write(f"| Median | {np.median(self.variances):.6f} |\n")
                f.write(f"| Min | {np.min(self.variances):.6f} |\n")
                f.write(f"| Max | {np.max(self.variances):.6f} |\n")
                f.write(f"| Std Dev | {np.std(self.variances):.6f} |\n\n")
                
                f.write("### Features by Variance Threshold\n\n")
                f.write("| Threshold | Count | Percentage |\n")
                f.write("|-----------|-------|------------|\n")
                for thresh in [0.001, 0.01, 0.1, 1.0]:
                    count = np.sum(self.variances < thresh)
                    percentage = (count / self.n_features) * 100
                    f.write(f"| < {thresh:.3f} | {count} | {percentage:.1f}% |\n")
                f.write("\n")
                
                # Correlation Analysis
                f.write("## 2. Correlation Analysis\n\n")
                f.write("Spearman correlation measures monotonic relationship with labels. Low correlation indicates weak predictive power.\n\n")
                
                abs_corr = np.abs(self.correlations)
                f.write("### Correlation Statistics (Absolute Values)\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Mean | {np.mean(abs_corr):.6f} |\n")
                f.write(f"| Median | {np.median(abs_corr):.6f} |\n")
                f.write(f"| Min | {np.min(abs_corr):.6f} |\n")
                f.write(f"| Max | {np.max(abs_corr):.6f} |\n")
                f.write(f"| Std Dev | {np.std(abs_corr):.6f} |\n\n")
                
                f.write("### Correlation Distribution\n\n")
                f.write("| Threshold | Count | Percentage |\n")
                f.write("|-----------|-------|------------|\n")
                for thresh in [0.01, 0.05, 0.10, 0.20]:
                    count = np.sum(abs_corr >= thresh)
                    percentage = (count / self.n_features) * 100
                    f.write(f"| >= {thresh:.2f} | {count} | {percentage:.1f}% |\n")
                f.write("\n")
                
                # Top Features
                f.write("### Top 15 Features by Correlation\n\n")
                f.write("| Rank | Feature | Correlation | P-value | Status |\n")
                f.write("|------|---------|-------------|---------|--------|\n")
                top_indices = np.argsort(abs_corr)[::-1][:15]
                for rank, idx in enumerate(top_indices, 1):
                    status = "✅ Selected" if (self.feature_mask is not None and self.feature_mask[idx]) else "❌ Rejected"
                    f.write(f"| {rank} | {self.feature_names[idx]} | {self.correlations[idx]:+.4f} | {self.p_values[idx]:.4f} | {status} |\n")
                f.write("\n")
                
                # Feature Selection
                if self.feature_mask is not None:
                    f.write("## 3. Feature Selection Results\n\n")
                    selected_count = np.sum(self.feature_mask)
                    rejected_count = self.n_features - selected_count
                    
                    f.write(f"**Selection Criteria:**\n")
                    f.write(f"- Variance >= {self.variance_threshold}\n")
                    f.write(f"- |Correlation| >= {self.correlation_threshold}\n\n")
                    
                    f.write("### Summary\n\n")
                    f.write("| Metric | Count | Percentage |\n")
                    f.write("|--------|-------|------------|\n")
                    f.write(f"| ✅ Selected | {selected_count} | {selected_count/self.n_features*100:.1f}% |\n")
                    f.write(f"| ❌ Rejected | {rejected_count} | {rejected_count/self.n_features*100:.1f}% |\n\n")
                    
                    # Rejection breakdown
                    variance_mask = self.variances >= self.variance_threshold
                    correlation_mask = np.abs(self.correlations) >= self.correlation_threshold
                    low_var_only = (~variance_mask) & correlation_mask
                    low_corr_only = variance_mask & (~correlation_mask)
                    both_low = (~variance_mask) & (~correlation_mask)
                    
                    f.write("### Rejection Breakdown\n\n")
                    f.write("| Reason | Count |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Low variance only | {np.sum(low_var_only)} |\n")
                    f.write(f"| Low correlation only | {np.sum(low_corr_only)} |\n")
                    f.write(f"| Both low | {np.sum(both_low)} |\n\n")
                
                # Recommendations
                f.write("## 4. Recommendations\n\n")
                
                if self.feature_mask is not None:
                    selected_count = np.sum(self.feature_mask)
                    reduction = ((self.n_features - selected_count) / self.n_features) * 100
                    
                    f.write(f"1. **Feature Reduction:** Remove {self.n_features - selected_count} low-quality features ({reduction:.1f}% reduction)\n")
                    f.write(f"2. **Expected Benefits:**\n")
                    f.write(f"   - Reduced noise and overfitting\n")
                    f.write(f"   - Faster training time\n")
                    f.write(f"   - Better generalization\n")
                    f.write(f"   - Estimated accuracy improvement: +10-15%\n\n")
                    
                    f.write(f"3. **Implementation:**\n")
                    f.write(f"   ```python\n")
                    f.write(f"   # Load feature mask\n")
                    f.write(f"   mask = np.load('data/cache/feature_selection_mask.npy')\n")
                    f.write(f"   \n")
                    f.write(f"   # Apply to features\n")
                    f.write(f"   X_filtered = X[:, mask]\n")
                    f.write(f"   ```\n\n")
                    
                    f.write(f"4. **Next Steps:**\n")
                    f.write(f"   - Retrain models with selected features\n")
                    f.write(f"   - Compare performance metrics\n")
                    f.write(f"   - Consider feature engineering for rejected features\n")
                    f.write(f"   - Monitor model performance over time\n\n")
                
                f.write("---\n\n")
                f.write("*Report generated by Feature Analysis Tool*\n")
            
            logger.info(f"✅ Report saved to {report_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False


def main():
    """Main entry point for the feature analyzer."""
    parser = argparse.ArgumentParser(
        description="Feature Analysis Tool for ML Feature Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run full analysis (variance + correlation)'
    )
    
    parser.add_argument(
        '--select',
        action='store_true',
        help='Select features and save mask'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate markdown report'
    )
    
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.01,
        help='Variance threshold for feature selection (default: 0.01)'
    )
    
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.05,
        help='Correlation threshold for feature selection (default: 0.05)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/cache/BTC-USDT_training_data.npz',
        help='Path to training data NPZ file'
    )
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not (args.analyze or args.select or args.report):
        parser.print_help()
        return 1
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(
        data_path=args.data_path,
        variance_threshold=args.variance_threshold,
        correlation_threshold=args.correlation_threshold
    )
    
    # Load data
    if not analyzer.load_data():
        logger.error("Failed to load data. Exiting.")
        return 1
    
    # Run analysis
    if args.analyze:
        analyzer.analyze_variance()
        analyzer.analyze_correlations()
    
    # Feature selection
    if args.select:
        if analyzer.variances is None or analyzer.correlations is None:
            logger.info("Running analysis before feature selection...")
            analyzer.analyze_variance()
            analyzer.analyze_correlations()
        
        analyzer.select_features()
        analyzer.save_feature_mask()
    
    # Generate report
    if args.report:
        if analyzer.variances is None or analyzer.correlations is None:
            logger.info("Running analysis before report generation...")
            analyzer.analyze_variance()
            analyzer.analyze_correlations()
        
        if analyzer.feature_mask is None and args.select:
            analyzer.select_features()
        
        analyzer.generate_report()
    
    logger.info("\n✅ Feature analysis completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
