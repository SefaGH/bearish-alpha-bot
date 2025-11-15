#!/usr/bin/env python3
"""
Master Test Runner for GEMMA Architecture Validation
Runs all validation tests and generates comprehensive report
"""
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime

# Enable ML for all tests
os.environ['ML_ENABLED'] = 'true'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from tests.validation import test_legacy_system
from tests.validation import test_manifest_manager
from tests.validation import test_feature_engineering
from tests.validation import test_component_compatibility


def run_test_suite(test_module, test_name):
    """Run a test suite and track timing"""
    print(f"\n{'='*60}")
    print(f"🔄 Running {test_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        success = test_module.run_all_tests()
        elapsed = time.time() - start_time
        return {
            'name': test_name,
            'passed': success,
            'elapsed_time': elapsed,
            'error': None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'name': test_name,
            'passed': False,
            'elapsed_time': elapsed,
            'error': str(e)
        }


def generate_test_report(results):
    """Generate comprehensive test report"""
    timestamp = datetime.now().isoformat()
    
    report = {
        "timestamp": timestamp,
        "version": "GEMMA Integration v1.0",
        "environment": "validation",
        "python_version": sys.version,
        "tests_executed": {},
        "summary": {
            "total_tests": len(results),
            "passed": 0,
            "failed": 0,
            "total_time": 0
        },
        "issues_found": [],
        "recommendations": [],
        "ready_for_production": False
    }
    
    # Process results
    for result in results:
        report["tests_executed"][result['name']] = {
            'passed': result['passed'],
            'elapsed_time': f"{result['elapsed_time']:.2f}s",
            'error': result['error']
        }
        
        if result['passed']:
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
            report["issues_found"].append(f"{result['name']}: {result['error'] or 'Test failed'}")
        
        report["summary"]["total_time"] += result['elapsed_time']
    
    # Determine production readiness
    report["ready_for_production"] = report["summary"]["failed"] == 0
    
    # Add recommendations
    if report["ready_for_production"]:
        report["recommendations"].append("All validation tests passed - system ready for production")
    else:
        report["recommendations"].append("Fix failing tests before production deployment")
    
    return report


def print_summary(report):
    """Print test summary to console"""
    print("\n" + "="*60)
    print("GEMMA INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Total Time: {report['summary']['total_time']:.2f}s")
    
    if report["ready_for_production"]:
        print("\n✅ SYSTEM READY FOR PRODUCTION")
    else:
        print("\n⚠️ SYSTEM NOT READY FOR PRODUCTION")
        print("\nIssues to resolve:")
        for issue in report["issues_found"]:
            print(f"  - {issue}")
    
    print("="*60)


def save_report(report):
    """Save report to file"""
    report_dir = Path("test_reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"gemma_validation_{timestamp_str}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {report_path}")


def main():
    """Main test execution"""
    print("="*60)
    print("🧪 GEMMA MANIFEST-DRIVEN ARCHITECTURE VALIDATION")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all test suites
    results = []
    
    results.append(run_test_suite(test_legacy_system, "Task 1: Legacy System Validation"))
    results.append(run_test_suite(test_manifest_manager, "Task 2: ManifestManager Functionality"))
    results.append(run_test_suite(test_feature_engineering, "Task 3: Feature Engineering Dynamic Loading"))
    results.append(run_test_suite(test_component_compatibility, "Task 4: Component Dimension Compatibility"))
    
    # Generate and display report
    report = generate_test_report(results)
    print_summary(report)
    save_report(report)
    
    # Exit with appropriate code
    sys.exit(0 if report["ready_for_production"] else 1)


if __name__ == "__main__":
    main()
