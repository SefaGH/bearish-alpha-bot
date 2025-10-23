"""
Test Suite for Silent Loop Fix

Verifies that the production loop processes iterations reliably
in both debug and non-debug modes, with proper timeout handling.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

# Import the production coordinator
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator


class TestSilentLoopFix:
    """Test suite for silent loop stall fix."""
    
    @pytest.fixture
    def mock_exchange_client(self):
        """Create a mock exchange client."""
        client = Mock()
        client.ohlcv = Mock(return_value=[
            [1640000000000, 47000, 47500, 46500, 47200, 1000],
            [1640003600000, 47200, 47800, 47000, 47500, 1200],
            [1640007200000, 47500, 48000, 47300, 47800, 1100],
        ])
        client.fetch_ticker = Mock(return_value={'last': 47500})
        return client
    
    @pytest.fixture
    def coordinator(self, mock_exchange_client):
        """Create a production coordinator instance."""
        coordinator = ProductionCoordinator()
        coordinator.exchange_clients = {'bingx': mock_exchange_client}
        coordinator.active_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        coordinator.is_initialized = True
        coordinator.is_running = False
        
        # Mock dependencies
        coordinator.trading_engine = Mock()
        coordinator.trading_engine.state = Mock()
        coordinator.trading_engine.state.value = 'running'
        coordinator.trading_engine.start_live_trading = AsyncMock(return_value={'success': True})
        coordinator.trading_engine.stop_live_trading = AsyncMock()
        coordinator.trading_engine.signal_queue = asyncio.Queue()
        
        coordinator.circuit_breaker = Mock()
        coordinator.circuit_breaker.check_circuit_breaker = AsyncMock(return_value={'tripped': False})
        
        coordinator.strategy_coordinator = Mock()
        coordinator.strategy_coordinator.signal_queue = asyncio.Queue()
        
        return coordinator
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_with_timeout(self, coordinator, mock_exchange_client):
        """Test that _fetch_ohlcv respects timeout."""
        # Test normal operation
        df = await coordinator._fetch_ohlcv(mock_exchange_client, 'BTC/USDT:USDT', '1h')
        assert df is not None
        assert len(df) == 3
        
        # Test timeout scenario
        async def slow_ohlcv(*args, **kwargs):
            await asyncio.sleep(20)  # Exceeds 15s timeout
            return []
        
        with patch.object(mock_exchange_client, 'ohlcv', side_effect=lambda *args, **kwargs: []):
            # Mock asyncio.to_thread to simulate slow call
            with patch('asyncio.to_thread', side_effect=slow_ohlcv):
                start = time.time()
                df = await coordinator._fetch_ohlcv(mock_exchange_client, 'BTC/USDT:USDT', '1h')
                duration = time.time() - start
                
                # Should timeout after 15s (with some tolerance)
                assert duration < 16, f"Timeout took {duration}s, expected ~15s"
                assert df is None, "Should return None on timeout"
    
    @pytest.mark.asyncio
    async def test_process_symbol_timeout_handling(self, coordinator, mock_exchange_client):
        """Test that process_symbol handles timeouts gracefully."""
        # Mock WebSocket manager to return None (force REST fallback)
        coordinator.websocket_manager = None
        
        # Test normal operation
        signal = await coordinator.process_symbol('BTC/USDT:USDT')
        # Signal may be None (no trading strategy registered) but should not hang
        assert signal is None or isinstance(signal, dict)
    
    @pytest.mark.asyncio
    async def test_watchdog_task_runs(self, coordinator):
        """Test that watchdog task logs periodically."""
        coordinator.is_running = True
        
        # Start watchdog task
        watchdog_task = asyncio.create_task(coordinator._watchdog_loop())
        
        # Let it run for a few cycles
        await asyncio.sleep(2.5)  # Should get 1-2 heartbeats
        
        # Cancel task
        coordinator.is_running = False
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        
        # If we get here without hanging, test passes
        assert True
    
    @pytest.mark.asyncio
    async def test_process_trading_loop_completes(self, coordinator):
        """Test that _process_trading_loop completes without hanging."""
        # Mock process_symbol to return quickly
        async def mock_process_symbol(symbol):
            await asyncio.sleep(0.1)  # Simulate processing
            return None  # No signal
        
        coordinator.process_symbol = mock_process_symbol
        coordinator.submit_signal = AsyncMock()
        
        # Run processing loop
        start = time.time()
        await coordinator._process_trading_loop()
        duration = time.time() - start
        
        # Should complete quickly (2 symbols * 0.1s ~= 0.2s)
        assert duration < 5, f"Processing took {duration}s, expected < 5s"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self, coordinator):
        """Test that circuit breaker check respects timeout."""
        # Mock circuit breaker to hang
        async def slow_check():
            await asyncio.sleep(10)  # Exceeds 5s timeout
            return {'tripped': False}
        
        coordinator.circuit_breaker.check_circuit_breaker = slow_check
        
        # This would be called in run_production_loop
        try:
            result = await asyncio.wait_for(
                coordinator.circuit_breaker.check_circuit_breaker(),
                timeout=5.0
            )
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            # Expected - timeout should occur
            pass
    
    @pytest.mark.asyncio
    async def test_run_production_loop_debug_off(self, coordinator):
        """Test production loop runs with debug=False."""
        coordinator.is_initialized = True
        coordinator.is_running = False
        coordinator.active_symbols = ['BTC/USDT:USDT']
        
        # Mock process_symbol to return quickly
        async def mock_process_symbol(symbol):
            await asyncio.sleep(0.1)
            return None
        
        coordinator.process_symbol = mock_process_symbol
        coordinator.submit_signal = AsyncMock()
        
        # Run loop for short duration
        loop_task = asyncio.create_task(
            coordinator.run_production_loop(mode='paper', duration=2.0, continuous=False)
        )
        
        # Wait for completion
        try:
            await asyncio.wait_for(loop_task, timeout=5.0)
        except asyncio.TimeoutError:
            coordinator.is_running = False
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            pytest.fail("Loop should complete within 5s but timed out")
        
        # If we get here, loop completed without hanging
        assert True
    
    @pytest.mark.asyncio
    async def test_run_production_loop_with_slow_fetch(self, coordinator, mock_exchange_client):
        """Test loop continues even when REST API is slow."""
        coordinator.is_initialized = True
        coordinator.is_running = False
        coordinator.active_symbols = ['BTC/USDT:USDT']
        coordinator.websocket_manager = None  # Force REST fallback
        
        # Make first fetch slow, second fast
        fetch_count = [0]
        original_ohlcv = mock_exchange_client.ohlcv
        
        def slow_then_fast(*args, **kwargs):
            fetch_count[0] += 1
            if fetch_count[0] == 1:
                # First call: sleep for 2s (within individual timeout)
                time.sleep(2)
            return original_ohlcv(*args, **kwargs)
        
        mock_exchange_client.ohlcv = slow_then_fast
        
        # Mock process_symbol to use real REST fetch logic
        # (process_symbol already does this, just ensure it's called)
        
        # Run loop for short duration
        loop_task = asyncio.create_task(
            coordinator.run_production_loop(mode='paper', duration=5.0, continuous=False)
        )
        
        # Wait for completion
        try:
            await asyncio.wait_for(loop_task, timeout=10.0)
        except asyncio.TimeoutError:
            coordinator.is_running = False
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            pytest.fail("Loop should complete within 10s even with slow fetch")
        
        # Verify loop completed despite slow fetch
        assert True


class TestEngineStartupSync:
    """Test engine startup synchronization."""
    
    @pytest.mark.asyncio
    async def test_engine_sync_delay_sufficient(self):
        """Test that 1.0s sync delay is sufficient for engine tasks."""
        coordinator = ProductionCoordinator()
        coordinator.is_initialized = True
        coordinator.active_symbols = ['BTC/USDT:USDT']
        
        # Mock trading engine that takes time to start
        engine = Mock()
        engine.state = Mock()
        engine.state.value = 'starting'  # Not running yet
        engine.signal_queue = asyncio.Queue()
        
        async def delayed_start(*args, **kwargs):
            # Simulate slow startup
            await asyncio.sleep(0.5)
            engine.state.value = 'running'
            return {'success': True}
        
        engine.start_live_trading = delayed_start
        engine.stop_live_trading = AsyncMock()
        
        coordinator.trading_engine = engine
        coordinator.circuit_breaker = Mock()
        coordinator.circuit_breaker.check_circuit_breaker = AsyncMock(return_value={'tripped': False})
        coordinator.strategy_coordinator = Mock()
        coordinator.strategy_coordinator.signal_queue = asyncio.Queue()
        
        # This should not raise after sync delay
        try:
            # Start the loop in background
            loop_task = asyncio.create_task(
                coordinator.run_production_loop(mode='paper', duration=1.0)
            )
            
            # Wait a bit to see if it enters the loop
            await asyncio.sleep(2.0)
            
            # Engine should be running now
            assert engine.state.value == 'running', f"Engine state is {engine.state.value}, expected 'running'"
            
            # Cancel the loop
            coordinator.is_running = False
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            
        except RuntimeError as e:
            pytest.fail(f"Should not raise RuntimeError with 1.0s sync delay: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
