from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Ensures that every strategy has a consistent interface for signal generation.
    """
    def __init__(self, strategy_name: str, config: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.config = config
        self.market_data_pipeline = None # Can be injected later

    def set_market_data_pipeline(self, pipeline: Any):
        """Inject the market data pipeline for data access."""
        self.market_data_pipeline = pipeline

    @abstractmethod
    async def generate_signal(self, symbol: str, ml_context: Optional[Dict] = None) -> Optional[Dict]:
        """
        The core method for a strategy to generate a trading signal.
        
        This method must be implemented by all subclasses. It should contain the
        primary logic for analyzing market data and deciding whether to issue a
        buy, sell, or hold signal.

        Args:
            symbol (str): The trading symbol to analyze (e.g., 'BTC/USDT').
            ml_context (Optional[Dict]): Optional machine learning context data.

        Returns:
            Optional[Dict]: A dictionary representing the trading signal if one is
                            generated, otherwise None.
        """
        pass
