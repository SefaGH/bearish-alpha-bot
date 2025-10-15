#!/usr/bin/env python3
"""
Test BingX Authentication Implementation

Tests for Phase 1 BingX authentication and connection fixes.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.bingx_authenticator import BingXAuthenticator
from core.ccxt_client import CcxtClient


def test_bingx_authenticator_initialization():
    """Test BingX authenticator initialization."""
    print("\n" + "="*60)
    print("TEST 1: BingX Authenticator Initialization")
    print("="*60)
    
    try:
        # Test with dummy credentials
        auth = BingXAuthenticator(
            api_key='test_key_123',
            secret_key='test_secret_456'
        )
        
        # Verify initialization
        if auth.api_key == 'test_key_123' and auth.secret_key == 'test_secret_456':
            print("✓ Authenticator initialized with credentials")
        else:
            print("✗ Credential storage failed")
            return False
        
        # Test timestamp generation
        timestamp = auth.get_timestamp_ms()
        if isinstance(timestamp, int) and timestamp > 0:
            print(f"✓ Timestamp generation working: {timestamp}")
        else:
            print("✗ Timestamp generation failed")
            return False
        
        print("\n✅ PASS: BingX Authenticator Initialization")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signature_generation():
    """Test HMAC-SHA256 signature generation."""
    print("\n" + "="*60)
    print("TEST 2: HMAC-SHA256 Signature Generation")
    print("="*60)
    
    try:
        auth = BingXAuthenticator(
            api_key='test_key',
            secret_key='test_secret'
        )
        
        # Test signature with known parameters
        params = {
            'symbol': 'BTC-USDT',
            'timestamp': 1234567890,
            'recvWindow': 5000
        }
        
        signature = auth.generate_signature(params)
        
        # Verify signature is hex string
        if isinstance(signature, str) and len(signature) == 64:
            print(f"✓ Signature generated: {signature[:16]}...")
            print(f"  Length: {len(signature)} characters (expected 64)")
        else:
            print("✗ Invalid signature format")
            return False
        
        # Test signature consistency (same params = same signature)
        signature2 = auth.generate_signature(params)
        if signature == signature2:
            print("✓ Signature is deterministic")
        else:
            print("✗ Signature is not consistent")
            return False
        
        # Test signature changes with different params
        params2 = {**params, 'timestamp': 1234567891}
        signature3 = auth.generate_signature(params2)
        if signature != signature3:
            print("✓ Signature changes with different parameters")
        else:
            print("✗ Signature does not change with different parameters")
            return False
        
        print("\n✅ PASS: Signature Generation")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_authenticated_request_preparation():
    """Test authenticated request preparation."""
    print("\n" + "="*60)
    print("TEST 3: Authenticated Request Preparation")
    print("="*60)
    
    try:
        auth = BingXAuthenticator(
            api_key='test_api_key',
            secret_key='test_secret_key'
        )
        
        # Test with no parameters
        result = auth.prepare_authenticated_request()
        
        # Verify structure
        if 'params' not in result or 'headers' not in result:
            print("✗ Missing params or headers in result")
            return False
        
        print("✓ Result contains 'params' and 'headers'")
        
        # Verify required parameters
        params = result['params']
        if 'timestamp' not in params:
            print("✗ Missing timestamp in params")
            return False
        if 'recvWindow' not in params:
            print("✗ Missing recvWindow in params")
            return False
        if 'signature' not in params:
            print("✗ Missing signature in params")
            return False
        
        print(f"✓ Required parameters present:")
        print(f"  - timestamp: {params['timestamp']}")
        print(f"  - recvWindow: {params['recvWindow']}")
        print(f"  - signature: {params['signature'][:16]}...")
        
        # Verify headers
        headers = result['headers']
        if 'X-BX-APIKEY' not in headers:
            print("✗ Missing X-BX-APIKEY header")
            return False
        if headers['X-BX-APIKEY'] != 'test_api_key':
            print("✗ Incorrect API key in header")
            return False
        if 'Content-Type' not in headers:
            print("✗ Missing Content-Type header")
            return False
        
        print("✓ Headers correctly set:")
        print(f"  - X-BX-APIKEY: {headers['X-BX-APIKEY']}")
        print(f"  - Content-Type: {headers['Content-Type']}")
        
        # Test with custom parameters
        custom_params = {'symbol': 'BTC-USDT', 'side': 'BUY'}
        result2 = auth.prepare_authenticated_request(custom_params)
        
        if 'symbol' in result2['params'] and result2['params']['symbol'] == 'BTC-USDT':
            print("✓ Custom parameters preserved")
        else:
            print("✗ Custom parameters not preserved")
            return False
        
        print("\n✅ PASS: Authenticated Request Preparation")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_symbol_conversion():
    """Test CCXT to BingX symbol format conversion."""
    print("\n" + "="*60)
    print("TEST 4: Symbol Format Conversion")
    print("="*60)
    
    try:
        auth = BingXAuthenticator(
            api_key='test_key',
            secret_key='test_secret'
        )
        
        # Test cases
        test_cases = [
            ('BTC/USDT:USDT', 'BTC-USDT'),
            ('ETH/USDT:USDT', 'ETH-USDT'),
            ('BTC/USDT', 'BTC-USDT'),
            ('SOL/USDT:USDT', 'SOL-USDT'),
            ('VST/USDT:USDT', 'VST-USDT'),
        ]
        
        all_passed = True
        for ccxt_symbol, expected_bingx in test_cases:
            result = auth.convert_symbol_to_bingx(ccxt_symbol)
            if result == expected_bingx:
                print(f"✓ {ccxt_symbol} → {result}")
            else:
                print(f"✗ {ccxt_symbol} → {result} (expected {expected_bingx})")
                all_passed = False
        
        if all_passed:
            print("\n✅ PASS: Symbol Conversion")
            return True
        else:
            print("\n❌ FAIL: Some conversions failed")
            return False
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ccxt_client_integration():
    """Test BingX authenticator integration with CcxtClient."""
    print("\n" + "="*60)
    print("TEST 5: CcxtClient Integration")
    print("="*60)
    
    try:
        # Test without credentials
        client1 = CcxtClient('bingx')
        if client1.bingx_auth is None:
            print("✓ Client without credentials has no authenticator")
        else:
            print("✗ Client without credentials should not have authenticator")
            return False
        
        # Test with credentials
        creds = {
            'apiKey': 'test_key',
            'secret': 'test_secret'
        }
        client2 = CcxtClient('bingx', creds)
        
        if client2.bingx_auth is not None:
            print("✓ Client with credentials has authenticator")
        else:
            print("✗ Client with credentials should have authenticator")
            return False
        
        # Verify authenticator properties
        if client2.bingx_auth.api_key == 'test_key':
            print("✓ API key correctly passed to authenticator")
        else:
            print("✗ API key not correctly passed")
            return False
        
        # Test authenticator methods are accessible
        try:
            timestamp = client2.bingx_auth.get_timestamp_ms()
            print(f"✓ Authenticator methods accessible: timestamp={timestamp}")
        except Exception as e:
            print(f"✗ Cannot access authenticator methods: {e}")
            return False
        
        print("\n✅ PASS: CcxtClient Integration")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_authenticated_methods_exist():
    """Test that authenticated methods exist on CcxtClient."""
    print("\n" + "="*60)
    print("TEST 6: Authenticated Methods Availability")
    print("="*60)
    
    try:
        creds = {
            'apiKey': 'test_key',
            'secret': 'test_secret'
        }
        client = CcxtClient('bingx', creds)
        
        # Check methods exist
        methods = [
            '_make_authenticated_bingx_request',
            'get_bingx_balance',
            'get_bingx_positions',
            'place_bingx_order'
        ]
        
        all_exist = True
        for method_name in methods:
            if hasattr(client, method_name):
                print(f"✓ Method exists: {method_name}")
            else:
                print(f"✗ Method missing: {method_name}")
                all_exist = False
        
        if all_exist:
            print("\n✅ PASS: All authenticated methods available")
            return True
        else:
            print("\n❌ FAIL: Some methods missing")
            return False
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all BingX authentication tests."""
    print("\n" + "="*60)
    print("BINGX AUTHENTICATION TEST SUITE")
    print("Phase 1: BingX Authentication & Connection Fix")
    print("="*60)
    
    tests = [
        test_bingx_authenticator_initialization,
        test_signature_generation,
        test_authenticated_request_preparation,
        test_symbol_conversion,
        test_ccxt_client_integration,
        test_authenticated_methods_exist,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} PASSED")
    print("="*60)
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
