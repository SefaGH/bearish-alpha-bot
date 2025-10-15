"""
Monitoring and Alerting System for Bearish Alpha Bot.

This module provides comprehensive monitoring, alerting, and analytics capabilities:
- Real-time web dashboard with WebSocket updates
- Multi-channel alert management (Telegram, Discord, Email, Webhook)
- Performance analytics and reporting
"""

from .dashboard import MonitoringDashboard
from .alert_manager import AlertManager, AlertPriority, AlertChannel, Alert
from .performance_analytics import PerformanceAnalytics

__all__ = [
    'MonitoringDashboard',
    'AlertManager',
    'AlertPriority',
    'AlertChannel',
    'Alert',
    'PerformanceAnalytics'
]
