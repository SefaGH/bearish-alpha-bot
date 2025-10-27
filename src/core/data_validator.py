"""
Data Consistency Validation Module.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS: Dict[str, int] = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200, '1d': 86400,
}

def validate_kline_timestamp(timestamp: int, timeframe: str, symbol: str) -> bool:
    """
    Validates if a kline's timestamp is correctly aligned with its timeframe.
    """
    if timeframe not in TIMEFRAME_SECONDS:
        return True # Bilinmeyen zaman dilimlerini atla

    timeframe_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    
    if timestamp % timeframe_ms != 0:
        logger.warning(
            f"⚠️ [DATA-CONSISTENCY] Timestamp mismatch for {symbol} on {timeframe}. "
            f"Timestamp {timestamp} is not divisible by {timeframe_ms}ms. Possible exchange data issue."
        )
        return False
        
    return True
