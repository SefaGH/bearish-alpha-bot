#!/usr/bin/env python3
"""
ML Component Verification for Live Context

Verifies ML components operate correctly in live context by analyzing logs.

Usage:
    python scripts/verify_ml_live.py <log_file>

Example:
    python scripts/verify_ml_live.py paper_trading_1hour.log
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

# Ensure src/ is importable when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FEATURE_COUNT = 42
DEFAULT_RL_STATE_SIZE = 42


def resolve_expected_dimensions() -> Tuple[int, int]:
    """Best-effort lookup of feature and RL state dimensions from manifest."""
    bundle_candidates = [
        Path("artifacts/gemma/final"),
        Path("artifacts/gemma/latest"),
        Path("artifacts/legacy"),
    ]

    try:
        from src.ml.manifest_manager import ManifestManager  # type: ignore

        manager = ManifestManager()

        for bundle in bundle_candidates:
            manifest_path = bundle / "manifest.json"
            if manifest_path.exists():
                manifest = manager.load_manifest(str(bundle))
                feature_count = manifest.get("feature_count", DEFAULT_FEATURE_COUNT)
                rl_state_size = manifest.get("rl_state_size", feature_count)
                return feature_count, rl_state_size

        manifest = manager.load_manifest()
        feature_count = manifest.get("feature_count", DEFAULT_FEATURE_COUNT)
        rl_state_size = manifest.get("rl_state_size", feature_count)
        return feature_count, rl_state_size

    except Exception:
        return DEFAULT_FEATURE_COUNT, DEFAULT_RL_STATE_SIZE


def verify_ml_operations(log_file: str) -> bool:
    """Verify ML components operate correctly in live context."""

    expected_feature_count, expected_rl_state_size = resolve_expected_dimensions()

    ml_operations: Dict[str, Dict[str, int]] = {
        'feature_engineering': defaultdict(int),
        'regime_predictor': defaultdict(int),
        'rl_agent': defaultdict(int),
        'price_predictor': defaultdict(int),
        'strategy_coordinator': defaultdict(int)
    }

    issues = []

    log_path = Path(log_file)
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return False

    with open(log_path, 'r', encoding='utf-8', errors='replace') as logfile:
        for line in logfile:
            if 'FeatureEngineeringPipeline' in line or 'extract_features' in line:
                match = re.search(r'Extracted\s+(\d+)', line)
                if match:
                    count = int(match.group(1))
                    if count == expected_feature_count:
                        ml_operations['feature_engineering']['correct_count'] += 1
                    else:
                        issues.append(
                            f"Wrong feature count: {count} (expected {expected_feature_count})"
                        )
                        ml_operations['feature_engineering']['wrong_count'] += 1

            if 'MLRegimePredictor' in line or 'regime' in line.lower():
                if 'prediction' in line or 'confidence' in line:
                    ml_operations['regime_predictor']['predictions'] += 1
                if 'ERROR' in line:
                    ml_operations['regime_predictor']['errors'] += 1
                    issues.append(f"Regime predictor error: {line[:100]}")

            if 'TradingRLAgent' in line or 'rl_agent' in line.lower():
                if 'action' in line:
                    ml_operations['rl_agent']['actions'] += 1
                if 'state_size' in line:
                    match = re.search(r'state_size[=:]\s*(\d+)', line)
                    if match:
                        state_size = int(match.group(1))
                        if state_size != expected_rl_state_size:
                            issues.append(
                                f"Wrong RL state size: {state_size} (expected {expected_rl_state_size})"
                            )
                            ml_operations['rl_agent']['wrong_state'] += 1

            if 'PricePrediction' in line or 'price_prediction' in line:
                if 'prediction' in line:
                    ml_operations['price_predictor']['predictions'] += 1
                if 'confidence' in line:
                    ml_operations['price_predictor']['confidence_scores'] += 1

            if 'StrategyCoordinator' in line:
                if 'signal' in line.lower():
                    ml_operations['strategy_coordinator']['signals'] += 1
                if 'ML enhancement' in line or 'ml_confidence' in line:
                    ml_operations['strategy_coordinator']['ml_enhancements'] += 1

    print("\n" + "=" * 60)
    print("ML COMPONENTS VERIFICATION REPORT")
    print("=" * 60)

    print("\n[Feature Engineering]")
    fe_stats = ml_operations['feature_engineering']
    print(f"  Correct Extractions ({expected_feature_count}): {fe_stats['correct_count']}")
    print(f"  Wrong Extractions: {fe_stats['wrong_count']}")

    print("\n[Regime Predictor]")
    rp_stats = ml_operations['regime_predictor']
    print(f"  Predictions Made: {rp_stats['predictions']}")
    print(f"  Errors: {rp_stats['errors']}")

    print("\n[RL Agent]")
    rl_stats = ml_operations['rl_agent']
    print(f"  Actions Taken: {rl_stats['actions']}")
    print(f"  Wrong State Size: {rl_stats['wrong_state']}")

    print("\n[Price Predictor]")
    pp_stats = ml_operations['price_predictor']
    print(f"  Predictions Made: {pp_stats['predictions']}")
    print(f"  Confidence Scores: {pp_stats['confidence_scores']}")

    print("\n[Strategy Coordinator]")
    sc_stats = ml_operations['strategy_coordinator']
    print(f"  Signals Generated: {sc_stats['signals']}")
    print(f"  ML Enhancements: {sc_stats['ml_enhancements']}")

    if issues:
        print("\n[ISSUES FOUND]")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("\nNo ML component issues detected")

    all_good = (
        fe_stats['wrong_count'] == 0 and
        rl_stats['wrong_state'] == 0 and
        rp_stats['errors'] == 0 and
        not issues
    )

    print("\n" + "=" * 60)
    if all_good:
        print("ML COMPONENTS WORKING CORRECTLY")
        Path("ml_components_verified.flag").touch()
    else:
        print("ML COMPONENTS HAVE ISSUES - REVIEW REQUIRED")

    return all_good


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_ml_live.py <log_file>")
        print("Example: python verify_ml_live.py paper_trading_1hour.log")
        sys.exit(1)

    LOG_FILE = sys.argv[1]
    SUCCESS = verify_ml_operations(LOG_FILE)
    sys.exit(0 if SUCCESS else 1)
