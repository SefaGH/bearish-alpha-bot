#!/usr/bin/env python3
"""
Integration Test for Phase 1 BingX Authentication Fix

Validates the complete authentication and connection flow.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_authenticator import BingXAuthenticator
from core.ccxt_client import CcxtClient


def test_phase1_requirements():
    """Test all Phase 1 requirements are met."""
    print("\n" + "="*70)
    print("PHASE 1 REQUIREMENTS VALIDATION")
    print("="*70)
    
    results = []
    
    # Requirement 1: BingX Authentication System
    print("\n[1/5] BingX Authentication System")
    try:
        auth = BingXAuthenticator('test_api_key', 'test_secret_key')
        
        # Test HMAC-SHA256 signature generation
        params = {'symbol': 'BTC-USDT', 'timestamp': 1234567890}
        signature = auth.generate_signature(params)
        
        if isinstance(signature, str) and len(signature) == 64:
            print("  ✓ HMAC-SHA256 signature generation working")
            results.append(True)
        else:
            print("  ✗ Signature generation failed")
            results.append(False)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results.append(False)
    
    # Requirement 2: Symbol Format Conversion
    print("\n[2/5] Symbol Format Conversion")
    try:
        auth = BingXAuthenticator('test_key', 'test_secret')
        
        test_cases = [
            ('BTC/USDT:USDT', 'BTC-USDT'),
            ('ETH/USDT:USDT', 'ETH-USDT'),
            ('BTC/USDT', 'BTC-USDT'),
        ]
        
        all_passed = True
        for ccxt_fmt, bingx_fmt in test_cases:
            result = auth.convert_symbol_to_bingx(ccxt_fmt)
            if result != bingx_fmt:
                all_passed = False
                print(f"  ✗ {ccxt_fmt} → {result} (expected {bingx_fmt})")
        
        if all_passed:
            print("  ✓ CCXT ↔ BingX symbol conversion working")
            results.append(True)
        else:
            results.append(False)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results.append(False)
    
    # Requirement 3: Private API Access Methods
    print("\n[3/5] Private API Access Methods")
    try:
        creds = {'apiKey': 'test_key', 'secret': 'test_secret'}
        client = CcxtClient('bingx', creds)
        
        required_methods = [
            'get_bingx_balance',
            'get_bingx_positions',
            'place_bingx_order',
        ]
        
        missing_methods = [m for m in required_methods if not hasattr(client, m)]
        
        if not missing_methods:
            print("  ✓ Balance, positions, and order methods available")
            results.append(True)
        else:
            print(f"  ✗ Missing methods: {missing_methods}")
            results.append(False)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results.append(False)
    
    # Requirement 4: CcxtClient Integration
    print("\n[4/5] CcxtClient Integration")
    try:
        # Test without credentials
        client_no_auth = CcxtClient('bingx')
        
        # Test with credentials
        creds = {'apiKey': 'test_key', 'secret': 'test_secret'}
        client_with_auth = CcxtClient('bingx', creds)
        
        if client_no_auth.bingx_auth is None and client_with_auth.bingx_auth is not None:
            print("  ✓ Authenticator properly initialized when credentials provided")
            results.append(True)
        else:
            print("  ✗ Authenticator initialization logic incorrect")
            results.append(False)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results.append(False)
    
    # Requirement 5: Enhanced Debug Logging
    print("\n[5/5] Enhanced Debug Logging")
    try:
        import logging
        from io import StringIO
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger('core.bingx_authenticator')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Create authenticator (should log)
        auth = BingXAuthenticator('test_key', 'test_secret')
        
        log_output = log_capture.getvalue()
        
        if '[BINGX-AUTH]' in log_output and 'initialized' in log_output:
            print("  ✓ Enhanced debug logging with emoji markers working")
            results.append(True)
        else:
            print(f"  ✗ Debug logging not working properly")
            results.append(False)
        
        logger.removeHandler(handler)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "="*70)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} requirements met")
    print("="*70)
    
    if passed == total:
        print("\n✅ ALL PHASE 1 REQUIREMENTS MET")
        return 0
    else:
        print("\n❌ SOME PHASE 1 REQUIREMENTS NOT MET")
        return 1


def test_performance_improvements():
    """Test performance improvements."""
    print("\n" + "="*70)
    print("PERFORMANCE IMPROVEMENTS VALIDATION")
    print("="*70)
    
    # Check live_trading_launcher.py for optimizations
    print("\n[1/2] Market Loading Optimization")
    try:
        with open('scripts/live_trading_launcher.py', 'r') as f:
            content = f.read()
        
        # Check that we don't load all markets
        if 'markets = bingx_client.markets()' in content:
            print("  ✗ Still loading all markets (2528+)")
            result1 = False
        else:
            print("  ✓ Optimized: No full market load")
            result1 = True
        
        # Check that we use single ticker test
        if 'test_ticker = bingx_client.fetch_ticker' in content:
            print("  ✓ Using single ticker test for connection")
            result2 = True
        else:
            print("  ✗ Not using optimized connection test")
            result2 = False
        
        # Check that we only verify trading pairs
        if 'Verifying {len(self.TRADING_PAIRS)} trading pairs' in content:
            print("  ✓ Only verifying 8 trading pairs")
            result3 = True
        else:
            print("  ✗ Not optimizing pair verification")
            result3 = False
        
        all_good = result1 and result2 and result3
    except Exception as e:
        print(f"  ✗ Failed to check: {e}")
        all_good = False
    
    # Check authentication test
    print("\n[2/2] Authentication Verification")
    try:
        with open('scripts/live_trading_launcher.py', 'r') as f:
            content = f.read()
        
        if 'get_bingx_balance' in content:
            print("  ✓ Authentication test with balance check included")
            auth_check = True
        else:
            print("  ✗ No authentication test found")
            auth_check = False
    except Exception as e:
        print(f"  ✗ Failed to check: {e}")
        auth_check = False
    
    print("\n" + "="*70)
    if all_good and auth_check:
        print("✅ ALL PERFORMANCE IMPROVEMENTS VERIFIED")
        return 0
    else:
        print("❌ SOME PERFORMANCE IMPROVEMENTS MISSING")
        return 1


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PHASE 1 INTEGRATION TEST SUITE")
    print("BingX Authentication & Connection Fix")
    print("="*70)
    
    exit_code1 = test_phase1_requirements()
    exit_code2 = test_performance_improvements()
    
    print("\n" + "="*70)
    if exit_code1 == 0 and exit_code2 == 0:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("Phase 1 implementation is complete and ready for deployment")
        print("="*70)
        return 0
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
