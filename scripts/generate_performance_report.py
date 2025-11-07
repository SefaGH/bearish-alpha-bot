"""
Performance Report Generator

This script generates comprehensive performance reports after model training
using the ModelPerformanceTracker utility.

It generates:
- Markdown summary report
- CSV export for analysis
- Console summary statistics

Usage:
    python scripts/generate_performance_report.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.utils.model_performance_tracker import ModelPerformanceTracker
from src.core.logger import setup_logger

# Setup logger
logger = setup_logger("performance-report", level=logging.INFO, log_to_file=True)


def generate_summary_statistics(tracker: ModelPerformanceTracker):
    """
    Generate and log summary statistics from training history.
    
    Args:
        tracker: ModelPerformanceTracker instance
    """
    logger.info("="*60)
    logger.info("📊 TRAINING SUMMARY STATISTICS")
    logger.info("="*60)
    
    all_trainings = tracker.get_all_trainings()
    
    if not all_trainings:
        logger.warning("No training sessions found in history.")
        return
    
    # Overall statistics
    logger.info(f"Total Training Sessions: {len(all_trainings)}")
    
    # Group by model type
    by_type = {}
    for training in all_trainings:
        model_type = training["model_type"]
        if model_type not in by_type:
            by_type[model_type] = []
        by_type[model_type].append(training)
    
    # Statistics per model type
    for model_type, trainings in by_type.items():
        logger.info(f"\n{model_type.upper()} Models:")
        logger.info(f"  - Total sessions: {len(trainings)}")
        
        # Calculate average metrics if available
        if trainings:
            latest = trainings[-1]
            logger.info(f"  - Latest training: {latest['timestamp']}")
            
            # Average training time
            avg_time = sum(t.get('training_time_seconds', 0) for t in trainings) / len(trainings)
            logger.info(f"  - Average training time: {avg_time:.2f}s")
            
            # Show latest metrics
            if latest.get('metrics'):
                logger.info(f"  - Latest metrics:")
                for metric_name, metric_value in latest['metrics'].items():
                    if isinstance(metric_value, float):
                        logger.info(f"      {metric_name}: {metric_value:.4f}")
                    else:
                        logger.info(f"      {metric_name}: {metric_value}")
        
        # Unique models trained
        unique_models = set(t['model_name'] for t in trainings)
        logger.info(f"  - Unique models: {len(unique_models)}")
        for model_name in sorted(unique_models):
            model_trainings = [t for t in trainings if t['model_name'] == model_name]
            logger.info(f"      {model_name}: {len(model_trainings)} session(s)")
    
    logger.info("\n" + "="*60)


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("📈 GENERATING PERFORMANCE REPORT")
    logger.info("="*60)
    
    try:
        # Initialize tracker
        tracker = ModelPerformanceTracker()
        
        # Generate summary statistics
        generate_summary_statistics(tracker)
        
        # Generate Markdown report
        logger.info("\n📝 Generating Markdown report...")
        md_report = tracker.generate_summary_report()
        logger.info(f"✅ Markdown report generated")
        
        # Export to CSV
        logger.info("\n📊 Exporting to CSV...")
        csv_path = tracker.export_to_csv()
        logger.info(f"✅ CSV export complete: {csv_path}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ PERFORMANCE REPORT GENERATION COMPLETE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
