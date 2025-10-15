#!/usr/bin/env python3
"""
Real-time monitoring dashboard for Bearish Alpha Bot.
Provides WebSocket-based live updates and performance metrics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import aiohttp
from aiohttp import web
import aiohttp_cors

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Web-based monitoring dashboard with real-time updates.
    """
    
    def __init__(self, port: int = 8080):
        """
        Initialize monitoring dashboard.
        
        Args:
            port: Port to run the dashboard server on
        """
        self.port = port
        self.app = web.Application()
        self.websockets: Set[web.WebSocketResponse] = set()
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'total_signals': 0,
            'total_trades': 0,
            'win_rate': 0.0,  # Stored as decimal (0.0-1.0), displayed as percentage in UI
            'total_pnl': 0.0,
            'open_positions': [],
            'recent_signals': [],
            'health_status': 'healthy',
            'last_update': datetime.now().isoformat()
        }
        self._setup_routes()
        self._setup_cors()
        self.runner = None
    
    def _setup_cors(self):
        """Setup CORS for cross-origin requests."""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def _setup_routes(self):
        """Setup web routes."""
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/api/metrics', self.metrics_handler)
        self.app.router.add_get('/api/positions', self.positions_handler)
        self.app.router.add_get('/api/performance', self.performance_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
    
    async def index_handler(self, request):
        """Serve the dashboard HTML."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bearish Alpha Bot - Monitoring Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0d1117; 
                    color: #c9d1d9;
                    padding: 20px;
                }
                .container { max-width: 1400px; margin: 0 auto; }
                .header { 
                    background: linear-gradient(135deg, #1f6feb, #388bfd);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                h1 { color: white; font-size: 28px; }
                .grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .card {
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 10px;
                    padding: 20px;
                }
                .card h2 { 
                    color: #58a6ff;
                    font-size: 14px;
                    text-transform: uppercase;
                    margin-bottom: 15px;
                }
                .metric { font-size: 32px; font-weight: bold; }
                .positive { color: #3fb950; }
                .negative { color: #f85149; }
                .neutral { color: #8b949e; }
                .status { 
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                }
                .status.healthy { background: #1f6feb; color: white; }
                .status.warning { background: #d29922; color: white; }
                .status.critical { background: #da3633; color: white; }
                .table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .table th {
                    text-align: left;
                    padding: 10px;
                    border-bottom: 1px solid #30363d;
                    color: #8b949e;
                    font-size: 12px;
                    text-transform: uppercase;
                }
                .table td {
                    padding: 10px;
                    border-bottom: 1px solid #21262d;
                }
                .live-indicator {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: #3fb950;
                    border-radius: 50%;
                    margin-right: 5px;
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><span class="live-indicator"></span>Bearish Alpha Bot - Live Monitoring</h1>
                    <div id="connection-status" style="margin-top: 10px; font-size: 14px;"></div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h2>Health Status</h2>
                        <div id="health-status" class="status healthy">HEALTHY</div>
                        <div style="margin-top: 10px; font-size: 14px;" id="uptime">Uptime: -</div>
                    </div>
                    
                    <div class="card">
                        <h2>Total P&L</h2>
                        <div id="total-pnl" class="metric neutral">$0.00</div>
                        <div style="margin-top: 10px; font-size: 14px;" id="pnl-change">Signals: <span id="total-signals">0</span></div>
                    </div>
                    
                    <div class="card">
                        <h2>Win Rate</h2>
                        <div id="win-rate" class="metric">0%</div>
                        <div style="margin-top: 10px; font-size: 14px;">
                            <span id="total-trades">0 trades</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Active Positions</h2>
                        <div id="active-positions" class="metric">0</div>
                        <div style="margin-top: 10px; font-size: 14px;" id="position-value">Value: $0.00</div>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="card" style="grid-column: span 2;">
                        <h2>Recent Signals</h2>
                        <table class="table" id="signals-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Reason</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="signals-tbody">
                                <tr><td colspan="5" style="text-align: center; color: #8b949e;">No signals yet</td></tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h2>System Info</h2>
                        <div style="font-size: 14px; line-height: 1.8;">
                            <div>Last Update: <span id="last-update">-</span></div>
                            <div style="margin-top: 10px;">Connected Clients: <span id="client-count">0</span></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let ws = null;
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = () => {
                        document.getElementById('connection-status').innerHTML = 
                            '<span style="color: #3fb950;">✓ Connected</span>';
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                    };
                    
                    ws.onclose = () => {
                        document.getElementById('connection-status').innerHTML = 
                            '<span style="color: #f85149;">✗ Disconnected - Reconnecting...</span>';
                        setTimeout(connectWebSocket, 5000);
                    };
                }
                
                function updateDashboard(data) {
                    // Update health status
                    if (data.health_status) {
                        const statusEl = document.getElementById('health-status');
                        statusEl.textContent = data.health_status.toUpperCase();
                        statusEl.className = 'status ' + data.health_status;
                    }
                    
                    // Update P&L
                    if (data.total_pnl !== undefined) {
                        const pnlEl = document.getElementById('total-pnl');
                        pnlEl.textContent = '$' + data.total_pnl.toFixed(2);
                        pnlEl.className = 'metric ' + (data.total_pnl >= 0 ? 'positive' : 'negative');
                    }
                    
                    // Update win rate
                    if (data.win_rate !== undefined) {
                        document.getElementById('win-rate').textContent = 
                            (data.win_rate * 100).toFixed(1) + '%';
                    }
                    
                    // Update trade count
                    if (data.total_trades !== undefined) {
                        document.getElementById('total-trades').textContent = 
                            data.total_trades + ' trades';
                    }
                    
                    // Update signal count
                    if (data.total_signals !== undefined) {
                        document.getElementById('total-signals').textContent = data.total_signals;
                    }
                    
                    // Update active positions
                    if (data.open_positions) {
                        document.getElementById('active-positions').textContent = 
                            data.open_positions.length;
                    }
                    
                    // Update signals table
                    if (data.recent_signals) {
                        updateSignalsTable(data.recent_signals);
                    }
                    
                    // Update uptime
                    if (data.start_time) {
                        const start = new Date(data.start_time);
                        const now = new Date();
                        const diff = now - start;
                        const hours = Math.floor(diff / 3600000);
                        const minutes = Math.floor((diff % 3600000) / 60000);
                        document.getElementById('uptime').textContent = 
                            `Uptime: ${hours}h ${minutes}m`;
                    }
                    
                    // Update last update time
                    if (data.last_update) {
                        const updateTime = new Date(data.last_update);
                        document.getElementById('last-update').textContent = 
                            updateTime.toLocaleTimeString();
                    }
                }
                
                function updateSignalsTable(signals) {
                    const tbody = document.getElementById('signals-tbody');
                    if (signals.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #8b949e;">No signals yet</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = signals.slice(0, 10).map(signal => `
                        <tr>
                            <td>${new Date(signal.timestamp).toLocaleTimeString()}</td>
                            <td>${signal.symbol || 'N/A'}</td>
                            <td class="${signal.side === 'buy' ? 'positive' : 'negative'}">${(signal.side || 'N/A').toUpperCase()}</td>
                            <td>${signal.reason || 'N/A'}</td>
                            <td>${signal.status || 'pending'}</td>
                        </tr>
                    `).join('');
                }
                
                // Connect on load
                connectWebSocket();
                
                // Initial data fetch
                fetch('/api/metrics')
                    .then(res => res.json())
                    .then(data => updateDashboard(data))
                    .catch(console.error);
                    
                // Refresh metrics every 5 seconds
                setInterval(() => {
                    fetch('/api/metrics')
                        .then(res => res.json())
                        .then(data => updateDashboard(data))
                        .catch(console.error);
                }, 5000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def metrics_handler(self, request):
        """Return current metrics as JSON."""
        return web.json_response(self.metrics)
    
    async def positions_handler(self, request):
        """Return current positions."""
        # Load from state file
        state_file = Path('data/state.json')
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    return web.json_response({
                        'open': state.get('open', {}),
                        'closed': state.get('closed', [])[-20:]  # Last 20 closed
                    })
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return web.json_response({'open': {}, 'closed': []})
    
    async def performance_handler(self, request):
        """Return performance metrics."""
        # Calculate performance metrics
        day_stats_file = Path('data/day_stats.json')
        if day_stats_file.exists():
            try:
                with open(day_stats_file) as f:
                    stats = json.load(f)
                    return web.json_response(stats)
            except Exception as e:
                logger.error(f"Error loading day stats: {e}")
        return web.json_response({})
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Send initial data
            await ws.send_json(self.metrics)
            
            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle incoming messages if needed
                    pass
                    
        except Exception as e:
            logger.error(f'WebSocket handler error: {e}')
        finally:
            self.websockets.discard(ws)
            
        return ws
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """
        Broadcast update to all connected clients.
        
        Args:
            data: Data to broadcast
        """
        if self.websockets:
            # Remove closed websockets
            closed_ws = set()
            for ws in self.websockets:
                if ws.closed:
                    closed_ws.add(ws)
                else:
                    try:
                        await ws.send_json(data)
                    except Exception as e:
                        logger.error(f"Error broadcasting to websocket: {e}")
                        closed_ws.add(ws)
            
            # Clean up closed connections
            self.websockets -= closed_ws
    
    def update_metrics(self, **kwargs):
        """
        Update metrics and broadcast to clients.
        
        Args:
            **kwargs: Metric values to update
                     win_rate should be provided as decimal (0.0-1.0), it will be
                     displayed as percentage in the UI
        """
        self.metrics.update(kwargs)
        self.metrics['last_update'] = datetime.now().isoformat()
        
        # Schedule broadcast
        asyncio.create_task(self.broadcast_update(self.metrics))
    
    async def start(self):
        """Start the dashboard server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"Dashboard running at http://0.0.0.0:{self.port}")
    
    async def stop(self):
        """Stop the dashboard server."""
        # Close all websockets
        for ws in self.websockets:
            await ws.close()
        self.websockets.clear()
        
        # Stop the server
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Dashboard stopped")
