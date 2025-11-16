#!/usr/bin/env python3
"""
Paper Trading Health Analysis

Analyzes paper trading log files for health metrics and issues.

Usage:
    python scripts/paper_trading_health.py <log_file> [duration_seconds]
    
Example:
    python scripts/paper_trading_health.py paper_trading_1hour.log 3600
"""
import sys
import re
from pathlib import Path
from datetime import datetime
import json


def analyze_paper_trading_log(log_file, duration_seconds):
    """Analyze paper trading log for health metrics"""
    
    health_metrics = {
        'total_lines': 0,
        'error_count': 0,
        'warning_count': 0,
        'feature_extraction_count': 0,
        'ml_prediction_count': 0,
        'signal_count': 0,
        'trade_count': 0,
        'dimension_errors': 0,
        'feature_counts': set(),
        'performance_issues': 0
    }
    
    error_patterns = {
        'dimension_mismatch': r'(shape|dimension|size|features).*mismatch|expected \d+ .* got \d+',
        'ml_error': r'(MLRegimePredictor|TradingRLAgent|FeatureEngineering).*Error',
        'connection_error': r'Connection.*error|timeout|refused',
        'memory_error': r'MemoryError|OutOfMemory'
    }
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return False

    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            health_metrics['total_lines'] += 1
            
            # Count errors and warnings
            if 'ERROR' in line or 'CRITICAL' in line:
                health_metrics['error_count'] += 1
                
                # Check for specific error types
                for error_type, pattern in error_patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        if error_type == 'dimension_mismatch':
                            health_metrics['dimension_errors'] += 1
            
            elif 'WARNING' in line:
                health_metrics['warning_count'] += 1
            
            # Count operations
            if 'Extracted' in line and 'features' in line:
                health_metrics['feature_extraction_count'] += 1
                # Extract feature count
                match = re.search(r'Extracted (\d+) features', line)
                if match:
                    health_metrics['feature_counts'].add(int(match.group(1)))
            
            if 'ML' in line and 'prediction' in line:
                health_metrics['ml_prediction_count'] += 1
            
            if 'SIGNAL' in line:
                health_metrics['signal_count'] += 1
            
            if 'TRADE' in line or 'ORDER' in line:
                health_metrics['trade_count'] += 1
            
            # Check for performance issues
            if 'took' in line and 'ms' in line:
                match = re.search(r'took (\d+\.?\d*)\s*ms', line)
                if match:
                    time_ms = float(match.group(1))
                    if time_ms > 1000:  # Over 1 second
                        health_metrics['performance_issues'] += 1
    
    # Calculate rates
    if duration_seconds > 0:
        health_metrics['errors_per_hour'] = (health_metrics['error_count'] / duration_seconds) * 3600
        health_metrics['predictions_per_minute'] = (health_metrics['ml_prediction_count'] / duration_seconds) * 60
        health_metrics['signals_per_minute'] = (health_metrics['signal_count'] / duration_seconds) * 60
    
    # Feature count analysis
    feature_counts_list = list(health_metrics['feature_counts'])
    health_metrics['feature_counts'] = feature_counts_list
    health_metrics['consistent_features'] = len(feature_counts_list) == 1 and (not feature_counts_list or feature_counts_list[0] == 42)
    
    # Generate health status
    if health_metrics['error_count'] == 0:
        health_status = 'HEALTHY'
    elif health_metrics['error_count'] < 10:
        health_status = 'WARNING'
    else:
        health_status = 'CRITICAL'
    
    health_metrics['status'] = health_status
    
    # Print report
    print("\n" + "="*60)
    print("PAPER TRADING HEALTH ANALYSIS")
    print("="*60)
    print(f"Log File: {log_file}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Status: {health_status}")
    
    print("\n[Operation Metrics]")
    print(f"  Total Log Lines: {health_metrics['total_lines']}")
    print(f"  Feature Extractions: {health_metrics['feature_extraction_count']}")
    print(f"  ML Predictions: {health_metrics['ml_prediction_count']}")
    print(f"  Signals Generated: {health_metrics['signal_count']}")
    print(f"  Trades Executed: {health_metrics['trade_count']}")
    
    print("\n[Issue Metrics]")
    print(f"  Total Errors: {health_metrics['error_count']}")
    print(f"  Total Warnings: {health_metrics['warning_count']}")
    print(f"  Dimension Errors: {health_metrics['dimension_errors']}")
    print(f"  Performance Issues: {health_metrics['performance_issues']}")
    
    print("\n[Rates]")
    if duration_seconds > 0:
        print(f"  Errors per Hour: {health_metrics['errors_per_hour']:.2f}")
        print(f"  Predictions per Minute: {health_metrics['predictions_per_minute']:.2f}")
        print(f"  Signals per Minute: {health_metrics['signals_per_minute']:.2f}")
    
    print("\n[Feature Analysis]")
    print(f"  Feature Counts Seen: {feature_counts_list}")
    consistency = "Yes" if health_metrics['consistent_features'] else "No"
    print(f"  Consistent Features (42): {consistency}")
    
    if health_metrics['dimension_errors'] > 0:
        print("\n[CRITICAL] Dimension mismatch errors detected!")
        print("   This must be fixed before production deployment.")
    
    # Save detailed report
    report_path = Path(f"paper_health_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(health_metrics, f, indent=2)
    print(f"\nDetailed report saved: {report_path}")
    
    return health_status == 'HEALTHY'


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python paper_trading_health.py <log_file> [duration_seconds]")
        print("Example: python paper_trading_health.py paper_trading_1hour.log 3600")
        sys.exit(1)
    
    log_file = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    
    healthy = analyze_paper_trading_log(log_file, duration)
    sys.exit(0 if healthy else 1)
