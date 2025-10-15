#!/usr/bin/env python3
"""
Advanced alert management with multi-channel support and smart grouping.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertChannel(Enum):
    """Alert delivery channels."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    timestamp: datetime
    priority: AlertPriority
    title: str
    message: str
    metadata: Dict[str, Any]
    channels: List[AlertChannel]
    
    def to_dict(self):
        """Convert alert to dictionary."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['priority'] = self.priority.value
        d['channels'] = [c.value for c in self.channels]
        return d


class AlertManager:
    """
    Manages alerts with deduplication, grouping, and multi-channel delivery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration dictionary with channel settings
        """
        self.config = config
        self.alert_history: List[Alert] = []
        self.alert_groups: Dict[str, List[Alert]] = {}
        self.channel_handlers = {}
        self._setup_channels()
        
        # Anti-spam settings
        self.rate_limits = {
            AlertPriority.CRITICAL: timedelta(minutes=1),
            AlertPriority.HIGH: timedelta(minutes=5),
            AlertPriority.MEDIUM: timedelta(minutes=15),
            AlertPriority.LOW: timedelta(hours=1),
            AlertPriority.INFO: timedelta(hours=4)
        }
        self.last_alert_times: Dict[str, datetime] = {}
    
    def _setup_channels(self):
        """Setup alert channel handlers."""
        # Telegram
        if self.config.get('telegram', {}).get('enabled'):
            try:
                from src.core.notify import Telegram
                self.channel_handlers[AlertChannel.TELEGRAM] = Telegram(
                    self.config['telegram']['bot_token'],
                    self.config['telegram']['chat_id']
                )
                logger.info("Telegram channel enabled")
            except Exception as e:
                logger.warning(f"Failed to setup Telegram: {e}")
        
        # Discord webhook
        if self.config.get('discord', {}).get('enabled'):
            self.channel_handlers[AlertChannel.DISCORD] = self._discord_handler
            logger.info("Discord channel enabled")
        
        # Email
        if self.config.get('email', {}).get('enabled'):
            self.channel_handlers[AlertChannel.EMAIL] = self._email_handler
            logger.info("Email channel enabled")
        
        # Generic webhook
        if self.config.get('webhook', {}).get('enabled'):
            self.channel_handlers[AlertChannel.WEBHOOK] = self._webhook_handler
            logger.info("Webhook channel enabled")
    
    async def send_alert(self, 
                        title: str,
                        message: str,
                        priority: AlertPriority = AlertPriority.INFO,
                        channels: Optional[List[AlertChannel]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        group_key: Optional[str] = None) -> bool:
        """
        Send an alert through specified channels.
        
        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority level
            channels: Channels to send through (default: all enabled)
            metadata: Additional metadata
            group_key: Key for grouping similar alerts
            
        Returns:
            True if alert was sent, False if rate-limited or failed
        """
        # Check rate limiting
        if not self._check_rate_limit(group_key or title, priority):
            logger.debug(f"Alert rate-limited: {title}")
            return False
        
        # Create alert
        alert = Alert(
            id=f"{datetime.now().timestamp()}_{hash(title)}",
            timestamp=datetime.now(),
            priority=priority,
            title=title,
            message=message,
            metadata=metadata or {},
            channels=channels or list(self.channel_handlers.keys())
        )
        
        # Add to history
        self.alert_history.append(alert)
        
        # Group if specified
        if group_key:
            if group_key not in self.alert_groups:
                self.alert_groups[group_key] = []
            self.alert_groups[group_key].append(alert)
            
            # Check if should send grouped alert
            if len(self.alert_groups[group_key]) >= 5:  # Group threshold
                await self._send_grouped_alert(group_key)
                self.alert_groups[group_key] = []
                return True
        
        # Send to channels
        success = await self._send_to_channels(alert)
        
        # Update rate limit tracking
        self.last_alert_times[group_key or title] = datetime.now()
        
        return success
    
    def _check_rate_limit(self, key: str, priority: AlertPriority) -> bool:
        """Check if alert should be rate-limited."""
        if key not in self.last_alert_times:
            return True
        
        time_since_last = datetime.now() - self.last_alert_times[key]
        rate_limit = self.rate_limits.get(priority, timedelta(hours=1))
        
        return time_since_last >= rate_limit
    
    async def _send_to_channels(self, alert: Alert) -> bool:
        """Send alert to specified channels."""
        results = []
        
        for channel in alert.channels:
            if channel not in self.channel_handlers:
                continue
                
            try:
                handler = self.channel_handlers[channel]
                
                if channel == AlertChannel.TELEGRAM:
                    # Format for Telegram
                    icon = self._get_priority_icon(alert.priority)
                    text = f"{icon} <b>{alert.title}</b>\n\n{alert.message}"
                    if alert.metadata:
                        text += f"\n\n<code>{json.dumps(alert.metadata, indent=2)}</code>"
                    handler.send(text)
                    results.append(True)
                    
                elif channel == AlertChannel.DISCORD:
                    await handler(alert)
                    results.append(True)
                    
                elif channel == AlertChannel.EMAIL:
                    await handler(alert)
                    results.append(True)
                    
                elif channel == AlertChannel.WEBHOOK:
                    await handler(alert)
                    results.append(True)
                    
            except Exception as e:
                logger.error(f"Failed to send alert to {channel}: {e}")
                results.append(False)
        
        return any(results)
    
    async def _send_grouped_alert(self, group_key: str):
        """Send a grouped alert for multiple similar alerts."""
        alerts = self.alert_groups.get(group_key, [])
        if not alerts:
            return
        
        # Get highest priority
        max_priority = max(alerts, key=lambda a: self._priority_value(a.priority)).priority
        
        # Create summary
        title = f"[GROUPED] {group_key} ({len(alerts)} alerts)"
        message = f"Received {len(alerts)} similar alerts:\n\n"
        
        for alert in alerts[-5:]:  # Show last 5
            message += f"• {alert.timestamp.strftime('%H:%M:%S')} - {alert.message[:50]}\n"
        
        # Send grouped alert
        await self.send_alert(
            title=title,
            message=message,
            priority=max_priority,
            metadata={'group_count': len(alerts), 'group_key': group_key}
        )
    
    def _get_priority_icon(self, priority: AlertPriority) -> str:
        """Get icon for priority level."""
        icons = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.MEDIUM: "⚡",
            AlertPriority.LOW: "ℹ️",
            AlertPriority.INFO: "📊"
        }
        return icons.get(priority, "📌")
    
    def _priority_value(self, priority: AlertPriority) -> int:
        """Get numeric value for priority comparison."""
        values = {
            AlertPriority.CRITICAL: 5,
            AlertPriority.HIGH: 4,
            AlertPriority.MEDIUM: 3,
            AlertPriority.LOW: 2,
            AlertPriority.INFO: 1
        }
        return values.get(priority, 0)
    
    async def _discord_handler(self, alert: Alert):
        """Send alert to Discord via webhook."""
        webhook_url = self.config['discord']['webhook_url']
        
        # Format embed
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": self._get_discord_color(alert.priority),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {"name": k, "value": str(v), "inline": True}
                for k, v in (alert.metadata or {}).items()
            ][:25]  # Discord limit
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                webhook_url,
                json={"embeds": [embed]},
                headers={"Content-Type": "application/json"}
            )
    
    def _get_discord_color(self, priority: AlertPriority) -> int:
        """Get Discord embed color for priority."""
        colors = {
            AlertPriority.CRITICAL: 0xFF0000,  # Red
            AlertPriority.HIGH: 0xFF8C00,      # Dark Orange
            AlertPriority.MEDIUM: 0xFFD700,    # Gold
            AlertPriority.LOW: 0x00BFFF,       # Sky Blue
            AlertPriority.INFO: 0x808080       # Gray
        }
        return colors.get(priority, 0x808080)
    
    async def _email_handler(self, alert: Alert):
        """Send alert via email (placeholder for future implementation)."""
        logger.info(f"Email alert: {alert.title}")
    
    async def _webhook_handler(self, alert: Alert):
        """Send alert to generic webhook."""
        webhook_url = self.config['webhook']['url']
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                webhook_url,
                json=alert.to_dict(),
                headers={"Content-Type": "application/json"}
            )
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of alerts in the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with alert statistics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff]
        
        # Count by priority
        priority_counts = {}
        for priority in AlertPriority:
            priority_counts[priority.value] = sum(
                1 for a in recent_alerts if a.priority == priority
            )
        
        return {
            'total_alerts': len(recent_alerts),
            'by_priority': priority_counts,
            'grouped_alerts': len(self.alert_groups),
            'time_window_hours': hours
        }
