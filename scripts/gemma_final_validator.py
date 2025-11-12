#!/usr/bin/env python3.11
# scripts/gemma_final_validator.py
"""
Final pre-deployment validation script for the GEMMA integration.
Ensures all components are in place and ready for production activation.
"""

import sys
import json
from pathlib import Path
import torch
import joblib

def validate_deployment() -> int:
    """Performs a full system validation and returns an exit code."""
    print("="*60)
    print("🚀 BEARISH ALPHA BOT - GEMMA DEPLOYMENT VALIDATION")
    print("="*60)

    errors = []
    warnings = []

    # 1. Python Version Check
    if sys.version_info[:2] != (3, 11):
        errors.append(f"CRITICAL: Python 3.11 is required by workflows. Found: {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print("✅ Python version is 3.11.")

    # 2. Main Model File Check
    model_path = Path('data/models/gemma/final/gemma_price.pt')
    if model_path.exists() and model_path.suffix == '.pt':
        try:
            torch.jit.load(str(model_path), map_location='cpu')
            print(f"✅ Main model file is valid: {model_path}")
        except Exception as e:
            errors.append(f"CRITICAL: Model file is corrupted or invalid: {model_path}. Error: {e}")
    else:
        errors.append(f"CRITICAL: Main model file not found: {model_path}")

    # 3. Feature Configuration File Check
    features_path = Path('features/gemma/selected/gemma_price_selected_82.json')
    if features_path.exists():
        with open(features_path) as f:
            data = json.load(f)
            count = data.get('count')
            if count == 82:
                print(f"✅ Feature config file is valid (82 features): {features_path}")
            else:
                errors.append(f"CRITICAL: Feature count in {features_path} is {count}, expected 82.")
    else:
        errors.append(f"CRITICAL: Feature config file not found: {features_path}")

    # 4. Scaler File Check
    scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
    if scaler_path.exists():
        try:
            joblib.load(scaler_path)
            print(f"✅ Scaler file is valid: {scaler_path}")
        except Exception as e:
            errors.append(f"CRITICAL: Scaler file is corrupted or invalid: {scaler_path}. Error: {e}")
    else:
        errors.append(f"CRITICAL: Scaler file not found: {scaler_path}")
    
    # 5. Shadow Mode Report Check
    shadow_report_path = Path('diagnostics/gemma/shadow_report.json')
    if shadow_report_path.exists():
        with open(shadow_report_path) as f:
            report = json.load(f)
            recommendation = report.get('recommendation', '')
            if 'SAFE_TO_DEPLOY' in recommendation:
                print(f"✅ Shadow mode validation passed with recommendation: '{recommendation}'")
            elif 'RECOMMENDED_WITH' in recommendation:
                warnings.append(f"Shadow mode validation passed with caution: '{recommendation}'")
            else:
                errors.append(f"CRITICAL: Shadow mode validation failed or requires tuning: '{recommendation}'")
    else:
        warnings.append("Shadow mode report not found. Run shadow validation before production deployment.")

    # --- Final Verdict ---
    print("\n" + "-"*60)
    if errors:
        print("\n❌ DEPLOYMENT BLOCKED. Critical errors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the critical errors before attempting to deploy.")
        return 1
    
    if warnings:
        print("\n⚠️ DEPLOYMENT READY WITH WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nProceed with caution and monitor the system closely after activation.")
    
    print("\n✅ GEMMA IS READY FOR PRODUCTION ACTIVATION!")
    print("You can now set GEMMA_ENABLED=true and restart the bot.")
    return 0

if __name__ == "__main__":
    sys.exit(validate_deployment())
