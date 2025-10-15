#!/usr/bin/env python3
"""Tests for monitoring scripts."""

import json
import os
import sys
from pathlib import Path
import tempfile
import shutil

import pytest

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with test data."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / 'data'
    data_dir.mkdir()
    
    # Create test state.json
    state = {
        "open": {
            "BTC-USDT": {
                "symbol": "BTC-USDT",
                "side": "long",
                "entry_price": 42000.0,
                "amount": 0.001
            }
        },
        "closed": [
            {
                "symbol": "ETH-USDT",
                "side": "long",
                "entry_price": 2200.0,
                "exit_price": 2300.0,
                "pnl": 10.5,
                "timestamp": "2025-10-15T20:00:00Z",
                "status": "closed"
            },
            {
                "symbol": "SOL-USDT",
                "side": "short",
                "entry_price": 100.0,
                "exit_price": 95.0,
                "pnl": 5.0,
                "timestamp": "2025-10-15T21:00:00Z",
                "status": "closed"
            },
            {
                "symbol": "BNB-USDT",
                "side": "long",
                "entry_price": 300.0,
                "exit_price": 290.0,
                "pnl": -10.0,
                "timestamp": "2025-10-15T21:30:00Z",
                "status": "closed"
            }
        ]
    }
    
    with open(data_dir / 'state.json', 'w') as f:
        json.dump(state, f, indent=2)
    
    # Create test day_stats.json
    day_stats = {
        "pnl": 5.5,
        "signals": 12,
        "date": "2025-10-15"
    }
    
    with open(data_dir / 'day_stats.json', 'w') as f:
        json.dump(day_stats, f, indent=2)
    
    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Cleanup
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.unit
def test_telegram_monitor_stats_collection(temp_data_dir):
    """Test that telegram_monitor correctly collects trading stats."""
    from telegram_monitor import get_trading_stats
    
    stats = get_trading_stats()
    
    assert stats['total_trades'] == 3
    assert stats['open_positions'] == 1
    assert stats['total_pnl'] == 5.5
    assert abs(stats['win_rate'] - 0.6666666666666666) < 0.0001
    assert stats['daily_pnl'] == 5.5
    assert stats['daily_signals'] == 12
    assert stats['status'] == 'running'


@pytest.mark.unit
def test_telegram_monitor_message_formatting(temp_data_dir):
    """Test that telegram_monitor formats messages correctly."""
    from telegram_monitor import get_trading_stats, format_telegram_report
    
    stats = get_trading_stats()
    message = format_telegram_report(stats)
    
    assert '📊' in message
    assert 'LIVE TRADING REPORT' in message
    assert 'Total Trades: 3' in message
    assert 'Open Positions: 1' in message
    assert '$5.50' in message
    assert '66.7%' in message
    assert 'Daily Stats' in message
    assert 'Signals: 12' in message


@pytest.mark.unit
def test_telegram_monitor_missing_files():
    """Test telegram_monitor handles missing files gracefully."""
    from telegram_monitor import get_trading_stats
    
    # Create empty temp dir
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        stats = get_trading_stats()
        
        # Should return default values
        assert stats['total_trades'] == 0
        assert stats['open_positions'] == 0
        assert stats['total_pnl'] == 0.0
        assert stats['win_rate'] == 0.0
        assert stats['status'] == 'running'
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


@pytest.mark.unit
def test_html_report_generation(temp_data_dir):
    """Test that HTML report is generated correctly."""
    from generate_html_report import generate_html_report
    
    generate_html_report()
    
    # Check that files were created
    report_path = Path('data/report.html')
    stats_path = Path('data/stats.json')
    
    assert report_path.exists()
    assert stats_path.exists()
    
    # Verify HTML content
    with open(report_path) as f:
        html = f.read()
    
    assert '<!DOCTYPE html>' in html
    assert 'Bearish Alpha Bot' in html
    assert 'Total P&L' in html
    assert 'Total Trades' in html
    assert 'Open Positions' in html
    assert 'Win Rate' in html
    assert '$5.50' in html
    assert '66.7%' in html
    
    # Verify JSON stats
    with open(stats_path) as f:
        stats = json.load(f)
    
    assert stats['total_trades'] == 3
    assert stats['open_positions'] == 1
    assert stats['total_pnl'] == 5.5
    assert abs(stats['win_rate'] - 0.6666666666666666) < 0.0001


@pytest.mark.unit
def test_html_report_empty_data():
    """Test HTML report generation with no trading data."""
    from generate_html_report import generate_html_report
    
    # Create empty temp dir
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Create data dir but no files
        Path('data').mkdir()
        
        generate_html_report()
        
        # Check that files were created with default values
        report_path = Path('data/report.html')
        stats_path = Path('data/stats.json')
        
        assert report_path.exists()
        assert stats_path.exists()
        
        # Verify JSON stats show zeros
        with open(stats_path) as f:
            stats = json.load(f)
        
        assert stats['total_trades'] == 0
        assert stats['open_positions'] == 0
        assert stats['total_pnl'] == 0
        assert stats['win_rate'] == 0
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


@pytest.mark.unit
def test_html_report_styling():
    """Test that HTML report includes proper styling."""
    from generate_html_report import generate_html_report
    
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        Path('data').mkdir()
        generate_html_report()
        
        with open('data/report.html') as f:
            html = f.read()
        
        # Check for CSS styling
        assert '<style>' in html
        assert 'background: #1a1a1a' in html
        assert 'color: #fff' in html
        assert '.positive { color: #4CAF50; }' in html
        assert '.negative { color: #f44336; }' in html
        assert 'table' in html
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
