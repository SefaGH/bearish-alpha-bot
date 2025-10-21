#!/usr/bin/env python3
"""
Standalone validation script for resource cleanup implementation.

This script validates that the code changes are correctly implemented
without requiring full dependency installation.

Run with: python tests/validate_cleanup_implementation.py
"""

import sys
import os
import ast
import inspect

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    print(f"Checking syntax: {os.path.basename(filepath)}...")
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        print(f"  ✅ Syntax valid")
        return True
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False


def check_method_exists(filepath, class_name, method_name, should_be_async=False):
    """Check if a method exists in a class."""
    print(f"Checking {class_name}.{method_name}() exists...")
    
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            is_async = isinstance(item, ast.AsyncFunctionDef)
                            
                            if should_be_async and not is_async:
                                print(f"  ❌ Method exists but is not async")
                                return False
                            
                            print(f"  ✅ Method exists{'(async)' if is_async else ''}")
                            return True
        
        print(f"  ❌ Method not found")
        return False
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_string_in_file(filepath, search_string, description):
    """Check if a string exists in a file."""
    print(f"Checking {description}...")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if search_string in content:
            print(f"  ✅ Found")
            return True
        else:
            print(f"  ❌ Not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print("🔍 RESOURCE CLEANUP IMPLEMENTATION VALIDATION")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # File paths
    ccxt_client_path = os.path.join('src', 'core', 'ccxt_client.py')
    launcher_path = os.path.join('scripts', 'live_trading_launcher.py')
    
    print("=" * 70)
    print("1. SYNTAX CHECKS")
    print("=" * 70)
    
    # Check syntax
    if check_file_syntax(ccxt_client_path):
        tests_passed += 1
    else:
        tests_failed += 1
    
    if check_file_syntax(launcher_path):
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    print("=" * 70)
    print("2. CcxtClient.close() METHOD")
    print("=" * 70)
    
    # Check CcxtClient.close()
    if check_method_exists(ccxt_client_path, 'CcxtClient', 'close', should_be_async=True):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check it calls exchange.close()
    if check_string_in_file(ccxt_client_path, 'await self.ex.close()', 
                           'CcxtClient.close() calls exchange.close()'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    print("=" * 70)
    print("3. OptimizedWebSocketManager.stop_streaming() METHOD")
    print("=" * 70)
    
    # Check stop_streaming exists
    if check_method_exists(launcher_path, 'OptimizedWebSocketManager', 
                          'stop_streaming', should_be_async=True):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check it closes ws_manager
    if check_string_in_file(launcher_path, 'self.ws_manager.close()',
                           'stop_streaming() calls ws_manager.close()'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    print("=" * 70)
    print("4. LiveTradingLauncher.cleanup() METHOD")
    print("=" * 70)
    
    # Check cleanup method exists
    if check_method_exists(launcher_path, 'LiveTradingLauncher', 
                          'cleanup', should_be_async=True):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check cleanup tracking variables
    if check_string_in_file(launcher_path, '_cleanup_done',
                           'Cleanup tracking variable _cleanup_done'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check idempotency
    if check_string_in_file(launcher_path, 'if self._cleanup_done:',
                           'Cleanup idempotency check'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check WebSocket cleanup
    if check_string_in_file(launcher_path, 'self.ws_optimizer.stop_streaming()',
                           'Cleanup calls stop_streaming()'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check exchange client cleanup
    if check_string_in_file(launcher_path, 'client.close()',
                           'Cleanup closes exchange clients'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check coordinator cleanup
    if check_string_in_file(launcher_path, 'self.coordinator.stop_system()',
                           'Cleanup stops coordinator'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check task cancellation
    if check_string_in_file(launcher_path, 'asyncio.all_tasks()',
                           'Cleanup cancels pending tasks'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    print("=" * 70)
    print("5. FINALLY BLOCKS WITH CLEANUP")
    print("=" * 70)
    
    # Check _run_once has cleanup in finally
    if check_string_in_file(launcher_path, 'await self.cleanup()',
                           '_run_once() calls cleanup()'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check main() has cleanup in finally
    if check_string_in_file(launcher_path, 'await launcher.cleanup()',
                           'main() calls launcher.cleanup()'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    print("=" * 70)
    print("6. EXIT CODES")
    print("=" * 70)
    
    # Check exit code 130 for Ctrl+C
    if check_string_in_file(launcher_path, 'return 130',
                           'Exit code 130 for KeyboardInterrupt'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Check exit code handling in main
    if check_string_in_file(launcher_path, 'exit_code = 130',
                           'Exit code set in main()'):
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print()
    
    if tests_failed == 0:
        print("✅ ALL VALIDATION CHECKS PASSED!")
        print("=" * 70)
        return 0
    else:
        print(f"❌ {tests_failed} VALIDATION CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
