"""
Data Aggregator for Cross-Exchange Normalization and Quality Management.

Provides multi-exchange data aggregation with quality scoring, consensus data
generation, and intelligent data source selection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Multi-Exchange Data Aggregator with quality management.
    
    Features:
    - Exchange-specific data normalization
    - Cross-exchange data aggregation with quality scores
    - Best data source selection using quality metrics
    - Weighted consensus data generation
    - OHLC integrity validation and outlier removal
    """
    
    # Outlier detection threshold (multiplier for IQR)
    OUTLIER_IQR_MULTIPLIER = 3.0
    
    def __init__(self, pipeline):
        """
        Initialize DataAggregator.
        
        Args:
            pipeline: MarketDataPipeline instance for data access
        """
        # Avoid circular import by checking type without importing
        if not hasattr(pipeline, 'exchanges') or not hasattr(pipeline, 'get_latest_ohlcv'):
            raise TypeError("pipeline must be a MarketDataPipeline instance")
        
        self.pipeline = pipeline
        self.quality_thresholds = {
            'min_candles': 50,
            'max_gap_ratio': 0.05,  # 5% missing candles acceptable
            'freshness_minutes': 60  # Data older than 1 hour considered stale
        }
        
        logger.info("🔄 DataAggregator initialized")
    
    def normalize_ohlcv_data(self, raw_data: List, exchange: str) -> pd.DataFrame:
        """
        Exchange-specific data normalization.
        
        Converts raw OHLCV list data to normalized pandas DataFrame with
        exchange-specific handling for data quirks.
        
        Args:
            raw_data: Raw OHLCV data [[timestamp, open, high, low, close, volume], ...]
            exchange: Exchange name for specific normalization rules
        
        Returns:
            Normalized DataFrame with timestamp index and OHLCV columns
        """
        if not raw_data:
            logger.warning(f"Empty raw_data provided for {exchange}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw_data, columns=cols)
        
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Exchange-specific normalization
        if exchange.lower() == 'bingx':
            # BingX may have specific quirks - handle them here
            logger.debug(f"Applying BingX-specific normalization")
            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif exchange.lower() in ['kucoinfutures', 'kucoin']:
            # KuCoin-specific normalization
            logger.debug(f"Applying KuCoin-specific normalization")
            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif exchange.lower() == 'binance':
            # Binance-specific normalization
            logger.debug(f"Applying Binance-specific normalization")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif exchange.lower() == 'bitget':
            # Bitget-specific normalization
            logger.debug(f"Applying Bitget-specific normalization")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif exchange.lower() == 'ascendex':
            # AscendEX-specific normalization
            logger.debug(f"Applying AscendEX-specific normalization")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        else:
            # Generic normalization for other exchanges
            logger.debug(f"Applying generic normalization for {exchange}")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN in critical columns
        df = df.dropna(subset=["open", "high", "low", "close"])
        
        # Set timestamp as index
        df = df.set_index("timestamp")
        
        # Sort by timestamp
        df = df.sort_index()
        
        logger.debug(f"Normalized {len(df)} candles from {exchange}")
        
        return df
    
    def aggregate_multi_exchange(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Cross-exchange data aggregation with quality scores.
        
        Fetches data from all available exchanges and calculates quality
        scores for each source.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe string (e.g., '30m', '1h', '4h', '1d')
        
        Returns:
            Dict with aggregated data and quality metrics:
            {
                'sources': {
                    'exchange_name': {
                        'data': DataFrame,
                        'quality_score': float,
                        'candle_count': int,
                        'freshness': str
                    }
                },
                'best_exchange': str,
                'total_sources': int
            }
        """
        logger.info(f"Aggregating data for {symbol} {timeframe} across exchanges")
        
        result = {
            'sources': {},
            'best_exchange': None,
            'total_sources': 0
        }
        
        # Collect data from all exchanges in the pipeline
        for exchange_name in self.pipeline.exchanges.keys():
            try:
                # Get data from pipeline storage
                df = self.pipeline.get_latest_ohlcv(symbol, timeframe, exchange=exchange_name)
                
                if df is None or df.empty:
                    logger.debug(f"No data available for {symbol} {timeframe} from {exchange_name}")
                    continue
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(df, exchange_name)
                
                # Determine freshness
                last_timestamp = df.index[-1] if not df.empty else None
                freshness = self._calculate_freshness(last_timestamp)
                
                result['sources'][exchange_name] = {
                    'data': df,
                    'quality_score': quality_score,
                    'candle_count': len(df),
                    'freshness': freshness
                }
                
                logger.debug(f"{exchange_name}: {len(df)} candles, quality={quality_score:.3f}, freshness={freshness}")
                
            except Exception as e:
                logger.warning(f"Failed to aggregate data from {exchange_name}: {e}")
                continue
        
        result['total_sources'] = len(result['sources'])
        
        # Find best exchange by quality score
        if result['sources']:
            best = max(result['sources'].items(), key=lambda x: x[1]['quality_score'])
            result['best_exchange'] = best[0]
            logger.info(f"Best exchange for {symbol} {timeframe}: {result['best_exchange']} "
                       f"(quality={best[1]['quality_score']:.3f})")
        
        return result
    
    def get_best_data_source(self, symbol: str, timeframe: str, 
                            exchanges: List[str] = None) -> Optional[str]:
        """
        Select most reliable data source using quality scoring.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            exchanges: Optional list of specific exchanges to consider.
                      If None, considers all available exchanges.
        
        Returns:
            Name of best exchange, or None if no suitable source found
        """
        logger.debug(f"Finding best data source for {symbol} {timeframe}")
        
        # Filter exchanges if specified
        available_exchanges = exchanges or list(self.pipeline.exchanges.keys())
        
        best_exchange = None
        best_score = -1
        
        for exchange_name in available_exchanges:
            if exchange_name not in self.pipeline.exchanges:
                logger.warning(f"Exchange {exchange_name} not in pipeline, skipping")
                continue
            
            try:
                df = self.pipeline.get_latest_ohlcv(symbol, timeframe, exchange=exchange_name)
                
                if df is None or df.empty:
                    continue
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(df, exchange_name)
                
                # Check minimum quality requirements
                if len(df) < self.quality_thresholds['min_candles']:
                    logger.debug(f"{exchange_name}: insufficient candles ({len(df)} < {self.quality_thresholds['min_candles']})")
                    continue
                
                # Check freshness
                last_timestamp = df.index[-1] if not df.empty else None
                freshness = self._calculate_freshness(last_timestamp)
                if freshness == 'expired':
                    logger.debug(f"{exchange_name}: data expired")
                    continue
                
                # Update best if this is better
                if quality_score > best_score:
                    best_score = quality_score
                    best_exchange = exchange_name
                    logger.debug(f"New best: {exchange_name} (score={quality_score:.3f})")
                
            except Exception as e:
                logger.warning(f"Error evaluating {exchange_name}: {e}")
                continue
        
        if best_exchange:
            logger.info(f"Best data source for {symbol} {timeframe}: {best_exchange} (score={best_score:.3f})")
        else:
            logger.warning(f"No suitable data source found for {symbol} {timeframe}")
        
        return best_exchange
    
    def get_consensus_data(self, symbol: str, timeframe: str, 
                          min_sources: int = 2) -> Optional[pd.DataFrame]:
        """
        Create weighted consensus data from multiple exchanges.
        
        Combines data from multiple exchanges using quality-weighted averaging
        to produce more reliable consensus OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            min_sources: Minimum number of sources required (default: 2)
        
        Returns:
            DataFrame with weighted consensus OHLCV data, or None if insufficient sources
        """
        logger.info(f"Generating consensus data for {symbol} {timeframe} (min_sources={min_sources})")
        
        # Aggregate data from all sources
        aggregated = self.aggregate_multi_exchange(symbol, timeframe)
        
        if aggregated['total_sources'] < min_sources:
            logger.warning(f"Insufficient sources for consensus: {aggregated['total_sources']} < {min_sources}")
            return None
        
        # Collect all DataFrames with their quality scores
        sources = []
        total_quality = 0
        
        for exchange_name, source_data in aggregated['sources'].items():
            df = source_data['data']
            quality = source_data['quality_score']
            
            # Validate and clean data
            df_clean = self._validate_and_clean(df)
            
            if not df_clean.empty:
                sources.append({
                    'exchange': exchange_name,
                    'data': df_clean,
                    'quality': quality
                })
                total_quality += quality
        
        if len(sources) < min_sources:
            logger.warning(f"Insufficient clean sources: {len(sources)} < {min_sources}")
            return None
        
        # Normalize quality scores to weights
        for source in sources:
            source['weight'] = source['quality'] / total_quality if total_quality > 0 else 1.0 / len(sources)
        
        logger.info(f"Creating consensus from {len(sources)} sources")
        for source in sources:
            logger.debug(f"  {source['exchange']}: weight={source['weight']:.3f}")
        
        # Find common timestamps across all sources
        common_index = sources[0]['data'].index
        for source in sources[1:]:
            common_index = common_index.intersection(source['data'].index)
        
        if len(common_index) == 0:
            logger.warning("No common timestamps found across sources")
            return None
        
        logger.debug(f"Found {len(common_index)} common timestamps")
        
        # Create consensus DataFrame
        consensus = pd.DataFrame(index=common_index)
        
        # Calculate weighted average for each OHLCV column
        for col in ['open', 'high', 'low', 'close', 'volume']:
            weighted_values = np.zeros(len(common_index))
            
            for source in sources:
                # Align source data with common index
                source_values = source['data'].loc[common_index, col].values
                weighted_values += source_values * source['weight']
            
            consensus[col] = weighted_values
        
        # Add metadata
        consensus.attrs['sources'] = [s['exchange'] for s in sources]
        consensus.attrs['weights'] = {s['exchange']: s['weight'] for s in sources}
        consensus.attrs['is_consensus'] = True
        
        logger.info(f"✅ Generated consensus data with {len(consensus)} candles from {len(sources)} sources")
        
        return consensus
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLC integrity validation and outlier removal.
        
        Validates OHLC relationships and removes statistical outliers.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Cleaned DataFrame with invalid data removed
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Validate OHLC relationships: high >= low, high >= open, high >= close, low <= open, low <= close
        invalid_mask = (
            (df_clean['high'] < df_clean['low']) |
            (df_clean['high'] < df_clean['open']) |
            (df_clean['high'] < df_clean['close']) |
            (df_clean['low'] > df_clean['open']) |
            (df_clean['low'] > df_clean['close']) |
            (df_clean['volume'] < 0)
        )
        
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} candles with invalid OHLC relationships")
            df_clean = df_clean[~invalid_mask]
        
        # Remove outliers using IQR method on close prices
        if len(df_clean) > 10:
            q1 = df_clean['close'].quantile(0.25)
            q3 = df_clean['close'].quantile(0.75)
            iqr = q3 - q1
            
            # Define outlier bounds using configurable multiplier
            # Standard is 1.5, but we use 3.0 for less aggressive filtering in crypto markets
            lower_bound = q1 - self.OUTLIER_IQR_MULTIPLIER * iqr
            upper_bound = q3 + self.OUTLIER_IQR_MULTIPLIER * iqr
            
            outlier_mask = (df_clean['close'] < lower_bound) | (df_clean['close'] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.debug(f"Removing {outlier_count} outlier candles")
                df_clean = df_clean[~outlier_mask]
        
        # Remove duplicates
        duplicate_count = df_clean.index.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Removing {duplicate_count} duplicate timestamps")
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        logger.debug(f"Validation complete: {len(df)} → {len(df_clean)} candles")
        
        return df_clean
    
    def _calculate_quality_score(self, df: pd.DataFrame, exchange_name: str) -> float:
        """
        Data quality scoring (0-1 scale).
        
        Evaluates data quality based on:
        - Completeness (number of candles vs expected)
        - Data gaps
        - Freshness
        - OHLC integrity
        - Volume consistency
        
        Args:
            df: DataFrame with OHLCV data
            exchange_name: Name of exchange for scoring context
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if df.empty:
            return 0.0
        
        score = 1.0
        
        # 1. Completeness score (30% weight)
        candle_count = len(df)
        expected_candles = self.quality_thresholds['min_candles']
        completeness = min(candle_count / expected_candles, 1.0)
        completeness_score = completeness * 0.3
        
        # 2. Gap analysis (25% weight)
        if len(df) > 1:
            # Calculate expected interval from first two candles
            time_diff = (df.index[1] - df.index[0]).total_seconds()
            if time_diff > 0:
                expected_count = int((df.index[-1] - df.index[0]).total_seconds() / time_diff) + 1
                actual_count = len(df)
                gap_ratio = 1.0 - abs(expected_count - actual_count) / expected_count
                gap_ratio = max(gap_ratio, 0.0)
                
                # Penalize if gap ratio exceeds threshold
                if 1.0 - gap_ratio > self.quality_thresholds['max_gap_ratio']:
                    gap_score = gap_ratio * 0.25
                else:
                    gap_score = 0.25
            else:
                gap_score = 0.25
        else:
            gap_score = 0.25
        
        # 3. Freshness score (20% weight)
        last_timestamp = df.index[-1] if not df.empty else None
        freshness = self._calculate_freshness(last_timestamp)
        
        if freshness == 'fresh':
            freshness_score = 0.20
        elif freshness == 'stale':
            freshness_score = 0.10
        else:  # expired
            freshness_score = 0.0
        
        # 4. OHLC integrity (15% weight)
        valid_ohlc = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        integrity_ratio = valid_ohlc.sum() / len(df)
        integrity_score = integrity_ratio * 0.15
        
        # 5. Volume consistency (10% weight)
        volume_valid = (df['volume'] >= 0).sum() / len(df)
        # Check for suspicious zero volume candles
        zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
        volume_score = (volume_valid * (1.0 - zero_volume_ratio * 0.5)) * 0.10
        
        # Calculate total score
        total_score = completeness_score + gap_score + freshness_score + integrity_score + volume_score
        
        logger.debug(f"{exchange_name} quality breakdown: "
                    f"completeness={completeness_score:.3f}, "
                    f"gaps={gap_score:.3f}, "
                    f"freshness={freshness_score:.3f}, "
                    f"integrity={integrity_score:.3f}, "
                    f"volume={volume_score:.3f}, "
                    f"total={total_score:.3f}")
        
        return round(total_score, 4)
    
    def _calculate_freshness(self, last_timestamp: Optional[pd.Timestamp]) -> str:
        """
        Calculate data freshness.
        
        Args:
            last_timestamp: Last timestamp in data
        
        Returns:
            Freshness category: 'fresh', 'stale', or 'expired'
        """
        if last_timestamp is None:
            return 'expired'
        
        now = datetime.now(timezone.utc)
        
        # Ensure last_timestamp is timezone-aware
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.tz_localize('UTC')
        
        age_minutes = (now - last_timestamp).total_seconds() / 60
        
        if age_minutes < self.quality_thresholds['freshness_minutes']:
            return 'fresh'
        elif age_minutes < self.quality_thresholds['freshness_minutes'] * 3:
            return 'stale'
        else:
            return 'expired'
