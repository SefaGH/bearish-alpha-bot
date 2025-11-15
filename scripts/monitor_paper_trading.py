#!/usr/bin/env python3
"""
Paper Trading Monitor

Monitors paper trading bot performance and generates reports.

Usage:
    python scripts/monitor_paper_trading.py <log_file> [duration_seconds]
    
Example:
    python scripts/monitor_paper_trading.py paper_trading_test_300s.log 300
"""
import sys
import time
import json
from datetime import datetime
from pathlib import Path


class PaperTradingMonitor:
    """Monitor paper trading bot performance"""
    
    def __init__(self, log_file="paper_trading_test.log"):
        self.log_file = log_file
        self.start_time = datetime.now()
        self.metrics = {
            'errors': [],
            'warnings': [],
            'trades': 0,
            'signals': 0,
            'feature_extractions': 0,
            'ml_predictions': 0,
            'memory_usage': [],
            'cpu_usage': []
        }
    
    def parse_log_line(self, line):
        """Parse log line for metrics"""
        if 'ERROR' in line or 'CRITICAL' in line:
            self.metrics['errors'].append(line.strip())
        elif 'WARNING' in line:
            self.metrics['warnings'].append(line.strip())
        elif 'TRADE' in line or 'ORDER' in line:
            self.metrics['trades'] += 1
        elif 'SIGNAL' in line:
            self.metrics['signals'] += 1
        elif 'Extracted' in line and 'features' in line:
            self.metrics['feature_extractions'] += 1
        elif 'ML' in line and ('prediction' in line or 'confidence' in line):
            self.metrics['ml_predictions'] += 1
    
    def monitor_resources(self):
        """Monitor system resources"""
        try:
            import psutil
            # Find bot process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'live_trading_launcher.py' in str(proc.info.get('cmdline', [])):
                    try:
                        # Get memory and CPU usage
                        memory_mb = proc.memory_info().rss / 1024 / 1024
                        cpu_percent = proc.cpu_percent(interval=1)
                        
                        self.metrics['memory_usage'].append(memory_mb)
                        self.metrics['cpu_usage'].append(cpu_percent)
                        
                        return memory_mb, cpu_percent
                    except:
                        pass
        except ImportError:
            pass  # psutil not available
        return None, None
    
    def monitor_log(self, duration_seconds=300):
        """Monitor log file for specified duration"""
        print(f"\n📊 Monitoring paper trading for {duration_seconds} seconds...")
        print("="*60)
        
        end_time = time.time() + duration_seconds
        last_report = time.time()
        report_interval = 30  # Report every 30 seconds
        
        # Open log file for reading
        log_path = Path(self.log_file)
        if not log_path.exists():
            print(f"⚠️ Log file not found yet: {self.log_file}")
            print("Waiting for log file to be created...")
        
        while time.time() < end_time:
            if log_path.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        # Read new lines
                        for line in f:
                            self.parse_log_line(line)
                except:
                    pass  # Continue monitoring even if file read fails
            
            # Monitor resources
            if time.time() - last_report > report_interval:
                memory, cpu = self.monitor_resources()
                
                # Print interim report
                elapsed = int(time.time() - self.start_time.timestamp())
                print(f"\n[{elapsed}s] Interim Report:")
                print(f"  Errors: {len(self.metrics['errors'])}")
                print(f"  Warnings: {len(self.metrics['warnings'])}")
                print(f"  Signals: {self.metrics['signals']}")
                print(f"  Trades: {self.metrics['trades']}")
                print(f"  ML Predictions: {self.metrics['ml_predictions']}")
                if memory:
                    print(f"  Memory: {memory:.1f} MB")
                    print(f"  CPU: {cpu:.1f}%")
                
                last_report = time.time()
            
            time.sleep(0.5)
    
    def generate_report(self):
        """Generate final monitoring report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'start_time': self.start_time.isoformat(),
            'duration_seconds': duration,
            'summary': {
                'total_errors': len(self.metrics['errors']),
                'total_warnings': len(self.metrics['warnings']),
                'total_signals': self.metrics['signals'],
                'total_trades': self.metrics['trades'],
                'total_ml_predictions': self.metrics['ml_predictions'],
                'feature_extractions': self.metrics['feature_extractions']
            },
            'performance': {
                'avg_memory_mb': sum(self.metrics['memory_usage']) / max(len(self.metrics['memory_usage']), 1),
                'max_memory_mb': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'avg_cpu_percent': sum(self.metrics['cpu_usage']) / max(len(self.metrics['cpu_usage']), 1),
                'max_cpu_percent': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
            },
            'errors': self.metrics['errors'][:10],  # First 10 errors
            'status': 'PASS' if len(self.metrics['errors']) == 0 else 'FAIL'
        }
        
        # Save report
        report_path = Path(f"paper_trading_report_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("PAPER TRADING TEST REPORT")
        print("="*60)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Status: {report['status']}")
        
        print("\n📊 Activity Summary:")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")
        
        print("\n💻 Performance Metrics:")
        for key, value in report['performance'].items():
            print(f"  {key}: {value:.2f}")
        
        if report['errors']:
            print("\n❌ Sample Errors:")
            for error in report['errors'][:3]:
                print(f"  - {error[:100]}...")
        
        print(f"\n📄 Full report saved: {report_path}")
        
        return report['status'] == 'PASS'


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_paper_trading.py <log_file> [duration_seconds]")
        print("Example: python monitor_paper_trading.py paper_trading_test_300s.log 300")
        sys.exit(1)
    
    log_file = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    monitor = PaperTradingMonitor(log_file)
    monitor.monitor_log(duration_seconds=duration)
    success = monitor.generate_report()
    sys.exit(0 if success else 1)
