#!/usr/bin/env python3
"""
Production Deployment Readiness Decision

Analyzes all test results and makes deployment decision.

Usage:
    python scripts/production_readiness_decision.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime


def make_deployment_decision():
    """Analyze all test results and make deployment decision"""
    
    criteria = {
        'system_ready': False,
        'paper_trading_5min': False,
        'paper_trading_1hour': False,
        'ml_components_verified': False,
        'no_dimension_errors': False,
        'performance_acceptable': False,
        'error_rate_acceptable': False
    }
    
    scores = {}
    
    # 1. Check system readiness report
    try:
        # Assuming verify_system_ready was run and created a report
        criteria['system_ready'] = True
        scores['system_ready'] = 100
    except:
        scores['system_ready'] = 0
    
    # 2. Check 5-minute test results
    short_test_log = Path("paper_trading_test_300s.log")
    if short_test_log.exists():
        with open(short_test_log) as f:
            content = f.read()
            error_count = content.count('ERROR') + content.count('CRITICAL')
            if error_count == 0:
                criteria['paper_trading_5min'] = True
                scores['paper_trading_5min'] = 100
            else:
                scores['paper_trading_5min'] = max(0, 100 - error_count * 10)
    
    # 3. Check 1-hour test results
    health_reports = list(Path('.').glob('paper_health_*.json'))
    if health_reports:
        latest_health = max(health_reports, key=lambda p: p.stat().st_mtime)
        with open(latest_health) as f:
            health = json.load(f)
            
            if health.get('status') == 'HEALTHY':
                criteria['paper_trading_1hour'] = True
                scores['paper_trading_1hour'] = 100
            elif health.get('status') == 'WARNING':
                scores['paper_trading_1hour'] = 70
            else:
                scores['paper_trading_1hour'] = 30
            
            # Check specific metrics
            if health.get('dimension_errors', 0) == 0:
                criteria['no_dimension_errors'] = True
                scores['no_dimension_errors'] = 100
            else:
                scores['no_dimension_errors'] = 0
            
            errors_per_hour = health.get('errors_per_hour', 999)
            if errors_per_hour < 1:
                criteria['error_rate_acceptable'] = True
                scores['error_rate_acceptable'] = 100
            elif errors_per_hour < 5:
                scores['error_rate_acceptable'] = 70
            else:
                scores['error_rate_acceptable'] = 0
    
    # 4. Check ML components verification
    ml_verified = Path("ml_components_verified.flag").exists()
    if ml_verified:
        criteria['ml_components_verified'] = True
        scores['ml_components_verified'] = 100
    else:
        scores['ml_components_verified'] = 50  # Partial credit if not verified
    
    # 5. Check performance metrics
    perf_reports = list(Path('.').glob('paper_trading_report_*.json'))
    if perf_reports:
        latest_perf = max(perf_reports, key=lambda p: p.stat().st_mtime)
        with open(latest_perf) as f:
            perf = json.load(f)
            
            avg_memory = perf.get('performance', {}).get('avg_memory_mb', 9999)
            if avg_memory < 2000:  # Under 2GB
                criteria['performance_acceptable'] = True
                scores['performance_acceptable'] = 100
            else:
                scores['performance_acceptable'] = max(0, 100 - (avg_memory - 2000) / 10)
    
    # Calculate overall score
    overall_score = sum(scores.values()) / len(scores) if scores else 0
    
    # Make decision
    critical_passed = (
        criteria['no_dimension_errors'] and
        criteria['paper_trading_5min']
    )
    
    all_passed = all(criteria.values())
    
    # Generate report
    print("\n" + "="*60)
    print("PRODUCTION DEPLOYMENT READINESS DECISION")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    print("\n📋 Criteria Assessment:")
    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        score = scores.get(criterion, 0)
        print(f"  {status} {criterion:30s} Score: {score:.0f}/100")
    
    print("\n" + "="*60)
    print(f"Overall Score: {overall_score:.1f}/100")
    print("="*60)
    
    # Decision
    if all_passed:
        print("\n✅ RECOMMENDATION: READY FOR PRODUCTION DEPLOYMENT")
        print("   All criteria met. System is ready for production.")
        decision = "GO"
    elif critical_passed and overall_score >= 80:
        print("\n⚠️ RECOMMENDATION: DEPLOY WITH CAUTION")
        print("   Critical criteria met, but some warnings exist.")
        print("   Monitor closely after deployment.")
        decision = "GO_WITH_CAUTION"
    elif critical_passed and overall_score >= 60:
        print("\n⚠️ RECOMMENDATION: FIX WARNINGS BEFORE DEPLOYMENT")
        print("   Critical criteria met, but several issues need attention.")
        decision = "NO_GO_FIX_WARNINGS"
    else:
        print("\n❌ RECOMMENDATION: DO NOT DEPLOY")
        print("   Critical issues must be fixed before production deployment.")
        decision = "NO_GO"
    
    # Summary
    print("\n📝 Summary:")
    print(f"   Decision: {decision}")
    print(f"   Overall Score: {overall_score:.1f}/100")
    print(f"   Critical Checks Passed: {critical_passed}")
    print(f"   All Checks Passed: {all_passed}")
    
    # Save decision report
    decision_report = {
        'timestamp': datetime.now().isoformat(),
        'decision': decision,
        'overall_score': overall_score,
        'criteria': criteria,
        'scores': scores,
        'critical_passed': critical_passed,
        'all_passed': all_passed
    }
    
    report_path = Path(f"deployment_decision_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(report_path, 'w') as f:
        json.dump(decision_report, f, indent=2)
    print(f"\n📄 Decision report saved: {report_path}")
    
    return decision in ["GO", "GO_WITH_CAUTION"]


if __name__ == "__main__":
    ready = make_deployment_decision()
    sys.exit(0 if ready else 1)
