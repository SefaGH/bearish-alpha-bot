#!/usr/bin/env python3
"""Generate HTML monitoring report."""

import json
from pathlib import Path
from datetime import datetime

def generate_html_report():
    """Generate HTML report from trading data."""
    
    # Load data
    stats = {'trades': [], 'open_positions': {}, 'total_pnl': 0}
    
    state_file = Path('data/state.json')
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            stats['trades'] = state.get('closed', [])[-20:]  # Last 20
            stats['open_positions'] = state.get('open', {})
            stats['total_pnl'] = sum(t.get('pnl', 0) for t in state.get('closed', []))
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bearish Alpha Bot - Monitoring Report</title>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #4CAF50; }}
        .stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{ 
            background: #2a2a2a; 
            padding: 20px; 
            border-radius: 8px;
            border: 1px solid #444;
        }}
        .metric {{ font-size: 32px; font-weight: bold; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
        }}
        th, td {{ 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #444;
        }}
        th {{ background: #2a2a2a; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Bearish Alpha Bot - Monitoring Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total P&L</h3>
                <div class="metric {'positive' if stats['total_pnl'] >= 0 else 'negative'}">
                    ${stats['total_pnl']:.2f}
                </div>
            </div>
            <div class="stat-card">
                <h3>Total Trades</h3>
                <div class="metric">{len(stats['trades'])}</div>
            </div>
            <div class="stat-card">
                <h3>Open Positions</h3>
                <div class="metric">{len(stats['open_positions'])}</div>
            </div>
            <div class="stat-card">
                <h3>Win Rate</h3>
                <div class="metric">
                    {(sum(1 for t in stats['trades'] if t.get('pnl', 0) > 0) / max(1, len(stats['trades'])) * 100):.1f}%
                </div>
            </div>
        </div>
        
        <h2>Recent Trades</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>P&L</th>
                <th>Status</th>
            </tr>"""
    
    for trade in stats['trades'][-10:]:
        pnl = trade.get('pnl', 0)
        html += f"""
            <tr>
                <td>{trade.get('timestamp', 'N/A')}</td>
                <td>{trade.get('symbol', 'N/A')}</td>
                <td>{trade.get('side', 'N/A')}</td>
                <td class="{'positive' if pnl >= 0 else 'negative'}">${pnl:.2f}</td>
                <td>{trade.get('status', 'closed')}</td>
            </tr>"""
    
    html += """
        </table>
    </div>
</body>
</html>"""
    
    # Save report
    report_path = Path('data/report.html')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML report saved to {report_path}")
    
    # Also save JSON stats
    stats_path = Path('data/stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'total_pnl': stats['total_pnl'],
            'total_trades': len(stats['trades']),
            'open_positions': len(stats['open_positions']),
            'win_rate': sum(1 for t in stats['trades'] if t.get('pnl', 0) > 0) / max(1, len(stats['trades']))
        }, f, indent=2)

if __name__ == '__main__':
    generate_html_report()
