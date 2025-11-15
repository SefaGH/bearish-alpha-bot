#!/usr/bin/env python3
"""
ML Component Verification for Live Context

Verifies ML components operate correctly in live context by analyzing logs.

Usage:
    python scripts/verify_ml_live.py <log_file>
    
Example:
    python scripts/verify_ml_live.py paper_trading_1hour.log
"""
import sys
import json
import re
from pathlib import Path
from collections import defaultdict


def verify_ml_operations(log_file):
    """Verify ML components operate correctly in live context"""
    
    ml_operations = {
        'feature_engineering': defaultdict(int),
        'regime_predictor': defaultdict(int),
        'rl_agent': defaultdict(int),
        'price_predictor': defaultdict(int),
        'strategy_coordinator': defaultdict(int)
    }
    
    issues = []
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return False
    
    with open(log_file, 'r') as f:
        for line in f:
            # Feature Engineering checks
            if 'FeatureEngineeringPipeline' in line or 'extract_features' in line:
                if 'Extracted 42' in line:
                    ml_operations['feature_engineering']['correct_count'] += 1
                elif 'Extracted' in line:
                    match = re.search(r'Extracted (\d+)', line)
                    if match and int(match.group(1)) != 42:
                        issues.append(f"Wrong feature count: {match.group(1)}")
                        ml_operations['feature_engineering']['wrong_count'] += 1
            
            # Regime Predictor checks
            if 'MLRegimePredictor' in line or 'regime' in line.lower():
                if 'prediction' in line or 'confidence' in line:
                    ml_operations['regime_predictor']['predictions'] += 1
                if 'ERROR' in line:
                    ml_operations['regime_predictor']['errors'] += 1
                    issues.append(f"Regime predictor error: {line[:100]}")
            
            # RL Agent checks
            if 'TradingRLAgent' in line or 'rl_agent' in line.lower():
                if 'action' in line:
                    ml_operations['rl_agent']['actions'] += 1
                if 'state_size' in line:
                    match = re.search(r'state_size[=:]\s*(\d+)', line)
                    if match and int(match.group(1)) != 42:
                        issues.append(f"Wrong RL state size: {match.group(1)}")
                        ml_operations['rl_agent']['wrong_state'] += 1
            
            # Price Predictor checks
            if 'PricePrediction' in line or 'price_prediction' in line:
                if 'prediction' in line:
                    ml_operations['price_predictor']['predictions'] += 1
                if 'confidence' in line:
                    ml_operations['price_predictor']['confidence_scores'] += 1
            
            # Strategy Coordinator checks
            if 'StrategyCoordinator' in line:
                if 'signal' in line.lower():
                    ml_operations['strategy_coordinator']['signals'] += 1
                if 'ML enhancement' in line or 'ml_confidence' in line:
                    ml_operations['strategy_coordinator']['ml_enhancements'] += 1
    
    # Generate report
    print("\n" + "="*60)
    print("ML COMPONENTS VERIFICATION REPORT")
    print("="*60)
    
    print("\n📊 Feature Engineering:")
    fe_stats = ml_operations['feature_engineering']
    print(f"  Correct Extractions (42): {fe_stats['correct_count']}")
    print(f"  Wrong Extractions: {fe_stats['wrong_count']}")
    
    print("\n🔮 Regime Predictor:")
    rp_stats = ml_operations['regime_predictor']
    print(f"  Predictions Made: {rp_stats['predictions']}")
    print(f"  Errors: {rp_stats['errors']}")
    
    print("\n🤖 RL Agent:")
    rl_stats = ml_operations['rl_agent']
    print(f"  Actions Taken: {rl_stats['actions']}")
    print(f"  Wrong State Size: {rl_stats['wrong_state']}")
    
    print("\n💰 Price Predictor:")
    pp_stats = ml_operations['price_predictor']
    print(f"  Predictions Made: {pp_stats['predictions']}")
    print(f"  Confidence Scores: {pp_stats['confidence_scores']}")
    
    print("\n📈 Strategy Coordinator:")
    sc_stats = ml_operations['strategy_coordinator']
    print(f"  Signals Generated: {sc_stats['signals']}")
    print(f"  ML Enhancements: {sc_stats['ml_enhancements']}")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues[:10]:  # First 10 issues
            print(f"  - {issue}")
    else:
        print("\n✅ No ML component issues detected")
    
    # Overall assessment
    all_good = (
        fe_stats['wrong_count'] == 0 and
        rp_stats['errors'] == 0 and
        rl_stats['wrong_state'] == 0 and
        len(issues) == 0
    )
    
    print("\n" + "="*60)
    if all_good:
        print("✅ ML COMPONENTS WORKING CORRECTLY")
        # Create flag file to indicate verification success
        Path("ml_components_verified.flag").touch()
    else:
        print("❌ ML COMPONENTS HAVE ISSUES - REVIEW REQUIRED")
    
    return all_good


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_ml_live.py <log_file>")
        print("Example: python verify_ml_live.py paper_trading_1hour.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    success = verify_ml_operations(log_file)
    sys.exit(0 if success else 1)
