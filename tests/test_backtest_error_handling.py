#!/usr/bin/env python3
"""
Test the improved error handling and logging in backtest scripts.
This test verifies that errors are properly caught, logged, and reported.
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

# Test cases
test_cases = [
    {
        "name": "Missing EXCHANGES env var",
        "env": {},
        "expected_in_stderr": ["EXCHANGES environment variable"],
        "should_fail": True
    },
    {
        "name": "Invalid credentials",
        "env": {
            "EXCHANGES": "kucoinfutures",
            "KUCOIN_KEY": "invalid",
            "KUCOIN_SECRET": "invalid",
            "KUCOIN_PASSWORD": "invalid",
            "BT_EXCHANGE": "kucoinfutures",
            "BT_SYMBOL": "BTC/USDT",
            "BT_LIMIT": "100",
        },
        "expected_in_stderr": ["Symbol validation failed", "kucoinfutures"],
        "should_fail": True
    },
]

def run_test(test_case):
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"Test: {test_case['name']}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env.update(test_case.get("env", {}))
    env["LOG_LEVEL"] = "INFO"  # Don't need DEBUG for tests
    
    # Run the backtest script
    result = subprocess.run(
        [sys.executable, "src/backtest/param_sweep.py"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/home/runner/work/bearish-alpha-bot/bearish-alpha-bot",
        timeout=30
    )
    
    # Check exit code
    if test_case["should_fail"]:
        if result.returncode == 0:
            print(f"❌ FAIL: Expected non-zero exit code, got {result.returncode}")
            return False
        else:
            print(f"✓ Correctly exited with code {result.returncode}")
    else:
        if result.returncode != 0:
            print(f"❌ FAIL: Expected exit code 0, got {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
        else:
            print(f"✓ Correctly exited with code 0")
    
    # Check for expected messages in stderr
    for expected in test_case.get("expected_in_stderr", []):
        if expected in result.stderr:
            print(f"✓ Found expected message: '{expected}'")
        else:
            print(f"❌ FAIL: Expected message not found: '{expected}'")
            print(f"STDERR:\n{result.stderr}")
            return False
    
    return True

def main():
    print("="*60)
    print("BACKTEST ERROR HANDLING TEST SUITE")
    print("="*60)
    
    os.chdir("/home/runner/work/bearish-alpha-bot/bearish-alpha-bot")
    
    results = []
    for test_case in test_cases:
        result = run_test(test_case)
        results.append((test_case["name"], result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
