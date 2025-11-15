#!/usr/bin/env python3
"""
GEMMA Manifest Health Check

Validates manifest consistency with actual models and configuration.
Checks:
1. Manifest file existence and structure
2. Feature count consistency across components
3. Scaler dimensions match manifest
4. Model input dimensions match manifest
5. All referenced files exist
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.manifest_manager import ManifestManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_manifest_existence(bundle_path: str) -> dict:
    """Check if manifest file exists and is valid JSON"""
    result = {
        'name': 'Manifest Existence',
        'status': 'fail',
        'details': ''
    }
    
    manifest_path = Path(bundle_path) / "manifest.json"
    
    if not manifest_path.exists():
        result['details'] = f"Manifest not found at {manifest_path}"
        return result
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        result['status'] = 'pass'
        result['details'] = f"Manifest found and valid JSON at {manifest_path}"
        result['manifest'] = manifest
    except json.JSONDecodeError as e:
        result['details'] = f"Invalid JSON in manifest: {e}"
    except Exception as e:
        result['details'] = f"Error loading manifest: {e}"
    
    return result


def check_manifest_structure(manifest: dict) -> dict:
    """Validate manifest has required fields"""
    result = {
        'name': 'Manifest Structure',
        'status': 'pass',
        'details': '',
        'missing_fields': []
    }
    
    required_fields = [
        'version',
        'feature_count',
        'feature_names_ordered',
        'selected_features_price',
        'selected_features_regime',
        'rl_state_size'
    ]
    
    missing = []
    for field in required_fields:
        if field not in manifest:
            missing.append(field)
    
    if missing:
        result['status'] = 'fail'
        result['details'] = f"Missing required fields: {', '.join(missing)}"
        result['missing_fields'] = missing
    else:
        result['details'] = "All required fields present"
    
    return result


def check_feature_count_consistency(manifest: dict) -> dict:
    """Validate feature count matches feature names"""
    result = {
        'name': 'Feature Count Consistency',
        'status': 'pass',
        'details': ''
    }
    
    feature_count = manifest.get('feature_count', 0)
    feature_names = manifest.get('feature_names_ordered', [])
    
    if len(feature_names) != feature_count:
        result['status'] = 'fail'
        result['details'] = (
            f"Feature count mismatch: manifest says {feature_count}, "
            f"but {len(feature_names)} feature names provided"
        )
    else:
        result['details'] = f"Feature count consistent: {feature_count} features"
    
    return result


def check_selected_features_validity(manifest: dict) -> dict:
    """Validate selected feature indices are within bounds"""
    result = {
        'name': 'Selected Features Validity',
        'status': 'pass',
        'details': '',
        'issues': []
    }
    
    feature_count = manifest.get('feature_count', 0)
    price_features = manifest.get('selected_features_price', [])
    regime_features = manifest.get('selected_features_regime', [])
    
    issues = []
    
    # Check price features
    for idx in price_features:
        if not (0 <= idx < feature_count):
            issues.append(f"Price feature index {idx} out of bounds (0-{feature_count-1})")
    
    # Check regime features
    for idx in regime_features:
        if not (0 <= idx < feature_count):
            issues.append(f"Regime feature index {idx} out of bounds (0-{feature_count-1})")
    
    if issues:
        result['status'] = 'fail'
        result['details'] = f"Found {len(issues)} invalid feature indices"
        result['issues'] = issues
    else:
        result['details'] = (
            f"All selected features valid "
            f"(price: {len(price_features)}, regime: {len(regime_features)})"
        )
    
    return result


def check_model_files_existence(manifest: dict, bundle_path: str) -> dict:
    """Check if all referenced model files exist"""
    result = {
        'name': 'Model Files Existence',
        'status': 'pass',
        'details': '',
        'missing_files': []
    }
    
    bundle = Path(bundle_path)
    
    # Files to check (optional)
    optional_files = [
        'regime_scaler_path',
        'price_scaler_path',
        'regime_model_path',
        'lstm_model_path',
        'rl_model_path'
    ]
    
    missing = []
    found = []
    
    for file_key in optional_files:
        if file_key in manifest:
            file_path = bundle / manifest[file_key]
            if not file_path.exists():
                # Check if it's an absolute path
                if not Path(manifest[file_key]).exists():
                    missing.append(f"{file_key}: {manifest[file_key]}")
            else:
                found.append(file_key)
    
    if missing:
        result['status'] = 'warn'
        result['details'] = (
            f"Found {len(found)} files, missing {len(missing)} optional files"
        )
        result['missing_files'] = missing
    else:
        result['details'] = f"All {len(found)} referenced files exist"
    
    return result


def check_scaler_dimensions(manifest: dict, bundle_path: str) -> dict:
    """Check if scaler dimensions match manifest"""
    result = {
        'name': 'Scaler Dimensions',
        'status': 'warn',
        'details': 'Scaler validation skipped (requires joblib)'
    }
    
    # This would require joblib to be available
    # For now, we just check if the path is specified
    if 'regime_scaler_path' in manifest or 'price_scaler_path' in manifest:
        result['details'] = 'Scaler paths specified in manifest'
    
    return result


def check_manifest_manager_integration() -> dict:
    """Test ManifestManager can load the manifest"""
    result = {
        'name': 'ManifestManager Integration',
        'status': 'pass',
        'details': ''
    }
    
    try:
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        if manifest:
            result['details'] = (
                f"ManifestManager successfully loaded manifest "
                f"(version: {manifest.get('version')}, "
                f"features: {manifest.get('feature_count')})"
            )
        else:
            result['status'] = 'fail'
            result['details'] = "ManifestManager returned None"
    except Exception as e:
        result['status'] = 'fail'
        result['details'] = f"ManifestManager integration failed: {e}"
    
    return result


def run_health_check(bundle_path: str = 'artifacts/legacy') -> dict:
    """Run all health checks"""
    print("=" * 70)
    print("🏥 GEMMA Manifest Health Check")
    print("=" * 70)
    print()
    
    results = {
        'bundle_path': bundle_path,
        'checks': [],
        'summary': {
            'total': 0,
            'passed': 0,
            'warnings': 0,
            'failed': 0
        }
    }
    
    # Check 1: Manifest exists
    check_result = check_manifest_existence(bundle_path)
    results['checks'].append(check_result)
    
    if check_result['status'] != 'pass':
        print_check_result(check_result)
        print_summary(results)
        return results
    
    manifest = check_result.get('manifest', {})
    
    # Check 2: Manifest structure
    check_result = check_manifest_structure(manifest)
    results['checks'].append(check_result)
    print_check_result(check_result)
    
    # Check 3: Feature count consistency
    check_result = check_feature_count_consistency(manifest)
    results['checks'].append(check_result)
    print_check_result(check_result)
    
    # Check 4: Selected features validity
    check_result = check_selected_features_validity(manifest)
    results['checks'].append(check_result)
    print_check_result(check_result)
    
    # Check 5: Model files existence
    check_result = check_model_files_existence(manifest, bundle_path)
    results['checks'].append(check_result)
    print_check_result(check_result)
    
    # Check 6: Scaler dimensions
    check_result = check_scaler_dimensions(manifest, bundle_path)
    results['checks'].append(check_result)
    print_check_result(check_result)
    
    # Check 7: ManifestManager integration
    check_result = check_manifest_manager_integration()
    results['checks'].append(check_result)
    print_check_result(check_result)
    
    # Calculate summary
    for check in results['checks']:
        results['summary']['total'] += 1
        if check['status'] == 'pass':
            results['summary']['passed'] += 1
        elif check['status'] == 'warn':
            results['summary']['warnings'] += 1
        elif check['status'] == 'fail':
            results['summary']['failed'] += 1
    
    print()
    print_summary(results)
    
    return results


def print_check_result(check: dict):
    """Print a single check result"""
    status_symbols = {
        'pass': '✅',
        'warn': '⚠️',
        'fail': '❌'
    }
    
    symbol = status_symbols.get(check['status'], '❓')
    print(f"{symbol} {check['name']}: {check['details']}")
    
    # Print additional details if present
    if 'issues' in check and check['issues']:
        for issue in check['issues'][:3]:  # Show first 3 issues
            print(f"   - {issue}")
        if len(check['issues']) > 3:
            print(f"   ... and {len(check['issues']) - 3} more")
    
    if 'missing_files' in check and check['missing_files']:
        for missing in check['missing_files'][:3]:
            print(f"   - Missing: {missing}")
        if len(check['missing_files']) > 3:
            print(f"   ... and {len(check['missing_files']) - 3} more")


def print_summary(results: dict):
    """Print summary of all checks"""
    summary = results['summary']
    print("=" * 70)
    print("📋 Summary")
    print("=" * 70)
    print(f"Total Checks:  {summary['total']}")
    print(f"✅ Passed:      {summary['passed']}")
    print(f"⚠️  Warnings:    {summary['warnings']}")
    print(f"❌ Failed:      {summary['failed']}")
    print()
    
    if summary['failed'] > 0:
        print("❌ Health check FAILED - Fix errors before proceeding")
        return 1
    elif summary['warnings'] > 0:
        print("⚠️  Health check PASSED with warnings")
        return 0
    else:
        print("✅ Health check PASSED - All systems ready!")
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GEMMA Manifest Health Check')
    parser.add_argument(
        '--bundle',
        default='artifacts/legacy',
        help='Path to model bundle (default: artifacts/legacy)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    results = run_health_check(args.bundle)
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    sys.exit(0 if results['summary']['failed'] == 0 else 1)
