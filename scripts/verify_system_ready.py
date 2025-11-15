#!/usr/bin/env python3
"""
Pre-Launch System Verification Script

Comprehensive pre-launch verification to ensure all systems are ready
before launching paper trading tests.

Usage:
    python scripts/verify_system_ready.py
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def verify_system_ready():
    """Comprehensive pre-launch verification"""
    checks = {
        'python_version': False,
        'manifest_exists': False,
        'manifest_valid': False,
        'config_exists': False,
        'exchange_configured': False,
        'ml_components_ready': False,
        'disk_space_available': False,
        'memory_available': False
    }
    
    errors = []
    warnings = []
    
    # 1. Python version check
    if sys.version_info >= (3, 11) and sys.version_info < (3, 12):
        checks['python_version'] = True
    else:
        errors.append(f"Python {sys.version} detected, need 3.11.x (not 3.12+)")
    
    # 2. Manifest check
    manifest_path = Path('artifacts/legacy/manifest.json')
    if manifest_path.exists():
        checks['manifest_exists'] = True
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                if manifest.get('feature_count') == 42:
                    checks['manifest_valid'] = True
                else:
                    errors.append(f"Manifest feature_count: {manifest.get('feature_count')} != 42")
        except Exception as e:
            errors.append(f"Manifest parse error: {e}")
    else:
        errors.append("Legacy manifest not found")
    
    # 3. Config check (look for any config file)
    config_files = ['config/config.yaml', 'config/config.example.yaml', 'config/config.debug.yaml']
    if any(Path(cf).exists() for cf in config_files):
        checks['config_exists'] = True
    else:
        errors.append("No config file found (checked: config.yaml, config.example.yaml, config.debug.yaml)")
    
    # 4. Exchange configuration
    if os.getenv('EXCHANGE_API_KEY') and os.getenv('EXCHANGE_SECRET'):
        checks['exchange_configured'] = True
    else:
        warnings.append("Exchange credentials not set (will use paper trading)")
        checks['exchange_configured'] = True  # OK for paper trading
    
    # 5. ML components check
    try:
        from ml.manifest_manager import ManifestManager
        from ml.feature_engineering import FeatureEngineeringPipeline
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        checks['ml_components_ready'] = True
    except Exception as e:
        errors.append(f"ML components error: {e}")
    
    # 6. Disk space check
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        if free_gb > 5:  # Need at least 5GB free
            checks['disk_space_available'] = True
        else:
            warnings.append(f"Low disk space: {free_gb:.1f}GB free")
    except Exception as e:
        warnings.append(f"Could not check disk space: {e}")
    
    # 7. Memory check
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        if available_gb > 2:  # Need at least 2GB available
            checks['memory_available'] = True
        else:
            warnings.append(f"Low memory: {available_gb:.1f}GB available")
    except ImportError:
        warnings.append("psutil not installed, skipping memory check")
        checks['memory_available'] = True  # Don't fail if psutil is not available
    except Exception as e:
        warnings.append(f"Could not check memory: {e}")
    
    # Generate report
    print("\n" + "="*60)
    print("SYSTEM READINESS CHECK")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python Version: {sys.version}")
    
    print("\n📋 Check Results:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    if errors:
        print("\n❌ ERRORS (Must Fix):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n⚠️ WARNINGS (Review):")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Overall status
    all_passed = all(checks.values())
    critical_passed = checks['python_version'] and checks['manifest_valid'] and checks['ml_components_ready']
    
    if all_passed:
        print("\n✅ SYSTEM READY FOR PAPER TRADING")
        return True
    elif critical_passed:
        print("\n⚠️ SYSTEM READY WITH WARNINGS")
        return True
    else:
        print("\n❌ SYSTEM NOT READY - FIX ERRORS FIRST")
        return False


if __name__ == "__main__":
    ready = verify_system_ready()
    sys.exit(0 if ready else 1)
