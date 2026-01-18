"""
Production Coordinator - Phase 3 Orchestration Layer
Manages the complete production trading system with all phases integrated.
"""

import logging
import asyncio
import inspect
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import time
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np  # For ML health checks

from core.logger import get_current_run_id

# Phase 1: Multi-Exchange Framework
from .multi_exchange import build_clients_from_env
from .ccxt_client import CcxtClient
from .execution_env import (
    get_bingx_env,
    get_execution_backend,
    format_mode_banner,
    get_trading_mode,
    get_prod_canary_0_evidence_dir,
    get_prod_canary_0_max_closed_trades,
    get_vst_fullbot_canary_side,
    get_vst_fullbot_canary_evidence_dir,
    get_vst_fullbot_canary_max_closed_trades,
    is_real_execution_enabled,
    is_prod_canary_0_cleanup_enabled,
    is_prod_canary_0_enabled,
    is_vst_fullbot_canary_cleanup_enabled,
    is_vst_fullbot_canary_enabled,
)

# Phase 2: Market Intelligence  
from core.market_regime import MarketRegimeAnalyzer
from .performance_monitor import PerformanceMonitor
from .websocket_manager import WebSocketManager
from config.live_trading_config import LiveTradingConfiguration
from config.risk_config import RiskConfiguration

# Performance Monitor için basit fallback
class RealTimePerformanceMonitor:
    """Basit performance monitor fallback."""
    def __init__(self):
        self.trades = []
        self.metrics = {}
        self.performance_history = {}
        self.optimization_feedback = {}
    
    def record_trade(self, trade_data):
        self.trades.append(trade_data)
    
    def get_metrics(self):
        return self.metrics
    
    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Production uyumluluğu için gerekli metod."""
        return {
            'strategy': strategy_name,
            'status': 'active',
            'metrics': {},
            'trade_count': 0
        }

from .order_manager import SmartOrderManager
from .position_manager import AdvancedPositionManager, PositionManagerPnlProvider

# Phase 3.1-3.3: Risk & Portfolio Management
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .strategy_coordinator import StrategyCoordinator
from .circuit_breaker import CircuitBreakerSystem
from .bingx_vst_balance import BingxVstBalanceError, create_vst_balance_client_from_ccxt_bingx_client

# Phase 3.4: Live Trading Components
from .live_trading_engine import LiveTradingEngine
from .market_data_pipeline import MarketDataPipeline
from .indicator_validator import IndicatorValidator


# Strategy imports - DÜZELTILDI
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip
from src.strategies.mean_reversion import VWAPMeanReversion

# Triple-fallback import strategy for maximum compatibility:
# 1. Direct utils import (when src/ is on sys.path)
# 2. Absolute src.utils import (when repo root is on sys.path)
# 3. Relative import (when imported as package module)
try:
    # Option 1: Direct import (scripts add src/ to sys.path)
    from utils.pnl_calculator import calculate_unrealized_pnl, calculate_pnl_percentage
except ModuleNotFoundError:
    try:
        # Option 2: Absolute import (repo root on sys.path)
        from src.utils.pnl_calculator import calculate_unrealized_pnl, calculate_pnl_percentage
    except ModuleNotFoundError as e:
        # Option 3: Relative import (package context)
        if e.name in ('src', 'src.utils', 'src.utils.pnl_calculator'):
            from ..utils.pnl_calculator import calculate_unrealized_pnl, calculate_pnl_percentage
        else:
            # Unknown module missing, re-raise
            raise
# Phase 4: ML Components (optional)
try:
    from ml.regime_predictor import RegimePredictor
    from ml.strategy_optimizer import StrategyOptimizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from ml.adapters.ppo_trading_adapter import PPOTradingAdapter
except ImportError:  # pragma: no cover - adapter optional in some envs
    PPOTradingAdapter = None

logger = logging.getLogger(__name__)
# ✅ EKLE: Logger seviyesini zorla INFO yap
logger.setLevel(logging.INFO)

# Constants
ML_HEALTH_CHECK_SYMBOL = "HEALTH_CHECK_BTC/USDT"  # Symbol for ML health checks


class ProductionCoordinator:
    """Coordinate all Phase 3 components for production deployment."""
    
    # --- DEĞİŞİKLİK 1: __init__ metodunu `config` alacak şekilde güncelle ---
    def __init__(self, config: Optional[Dict] = None):
        """Initialize production coordinator."""
        # Phase 3 components (will be initialized)
        self.websocket_manager = None  # Phase 3.1
        self.risk_manager = None       # Phase 3.2
        self.portfolio_manager = None  # Phase 3.3
        self.trading_engine = None     # Phase 3.4
        self.strategy_coordinator = None
        self.circuit_breaker = None
        
        # Phase 2 components
        self.market_regime_analyzer = None  # Phase 2
        self.performance_monitor = None     # Phase 2
        
        # Phase 1 components
        self.exchange_clients = {}          # Phase 1
        
        # ML integration (Phase 4)
        self.ml_integration = None          # Prevents AttributeError during ML initialization
        self.ppo_adapter = None
        
        # Registered strategies
        self.strategies = {}  # strategy_name -> strategy_instance
        self.strategy_capabilities = {}  # strategy_name -> {supports_regime_data, is_async}
        
        # System state
        self.is_running = False
        self.is_initialized = False
        self.emergency_stop_triggered = False
        self.loop_interval = 30  # Ana döngü bekleme süresi
        self.active_symbols = []  # Takip edilen semboller
        self.processed_symbols_count = 0  # İşlenen sembol sayacı
        
        # Signal lifecycle tracking
        self.signal_lifecycle = {}  # signal_id -> {stage, timestamp, details}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._rl_telemetry_task: Optional[asyncio.Task] = None
        self._recheck_wakeup_event: Optional[asyncio.Event] = asyncio.Event()
        
        # --- DEĞİŞİKLİK 2: Kendi config'ini yüklemek yerine dışarıdan gelen config'i kullan ---
        # Eğer dışarıdan bir config gelmezse, eski davranışa geri dön (güvenlik için)
        if config:
            self.config = config
            logger.info("ProductionCoordinator initialized with provided configuration.")
        else:
            self.config = LiveTradingConfiguration.load()
            logger.warning("ProductionCoordinator initialized by loading its own configuration. (Legacy mode)")

        rl_cfg_dbg = (self.config.get('ml', {}) or {}).get('reinforcement_learning', {})
        logger.info(
            "🧪 [PPO-CONFIG] enabled=%s | symbols=%s",
            rl_cfg_dbg.get('ppo_enabled'),
            rl_cfg_dbg.get('ppo_symbols'),
        )

        # Debug ayarını config'den oku
        self.debug_logging = self.config.get('debug', {}).get('strategy_logging', False)
        
        # Market regime analyzer başlat
        self.market_regime_analyzer = None #MarketRegimeAnalyzer()
        
        logger.info("ProductionCoordinator created")

    async def verify_indicator_health(self) -> bool:
        """
        Perform real-time indicator health check during trading loop.
        Called every 5 minutes to ensure indicators remain valid.
        
        Returns:
            bool: True if all indicators healthy, False otherwise
        """
        logger.info("🏥 [HEALTH-CHECK] Performing indicator health check...")
        
        try:
            # --- NİHAİ DÜZELTME BAŞLANGICI ---
            # HATA: Validator'a, 'get_latest_ohlcv' metodu olmayan websocket_manager nesnesi veriliyordu.
            # DOĞRUSU: Validator, 'get_latest_ohlcv' metoduna sahip olan ve 
            # websocket_manager'ın içinde bulunan 'collector' nesnesine ihtiyaç duyar.
            if not self.websocket_manager or not hasattr(self.websocket_manager, 'collector'):
                 logger.error("❌ [HEALTH-CHECK] WebSocket manager or collector not available!")
                 return False
            
            validator = IndicatorValidator(self.websocket_manager.collector, config=self.config)
            # --- NİHAİ DÜZELTME SONU ---
            
            # 1. Gerekli zaman dilimlerini config'den alalım.
            ws_config = self.config.get('websocket', {})
            timeframes = ws_config.get('stream_timeframes', ['1m', '5m', '15m', '30m', '1h', '4h'])

            # 2. Doğru metot adını ('validate_all') ve doğru parametreleri ('timeframes') kullanalım.
            results = await validator.validate_all(
                symbols=self.active_symbols,
                timeframes=timeframes
            )

            # validate_all metodu doğrudan {symbol: result} formatında bir sözlük döndürüyor.
            valid_count = sum(1 for res in results.values() if res.get('status') == 'OK')
            all_valid = valid_count == len(self.active_symbols)

            if not all_valid:
                logger.warning("⚠️  [HEALTH-CHECK] Some indicators unhealthy")
                
                # --- DETAYLI LOGLAMA KISMI (KORUNUYOR) ---
                # Log unhealthy indicators
                for symbol, result in results.items():
                    # 'status' != 'OK' kontrolü, result sözlüğünün yapısıyla daha uyumludur.
                    if result.get('status') != 'OK':
                        reason = result.get('reason', 'Unknown error')
                        logger.warning(f"   - {symbol}: {reason}")
                        # Eğer 'errors' adında daha detaylı bir liste varsa onu da loglayalım.
                        if 'errors' in result and isinstance(result['errors'], list):
                             for error_detail in result['errors']:
                                 logger.warning(f"     - Detail: {error_detail}")

                return False
            
            logger.info("✅ [HEALTH-CHECK] All indicators healthy")
            return True
            
        except Exception as e:
            # Hatanın tam olarak nerede ve neden olduğunu anlamak için exc_info=True ekliyoruz.
            logger.error(f"❌ [HEALTH-CHECK] Health check failed critically: {e}", exc_info=True)
            return False

    def _normalize_symbol_for_ws(self, symbol: str) -> str:
        """
        Normalize a spot-like CCXT symbol to futures format used by WebSocket collector.
        'BTC/USDT' -> 'BTC/USDT:USDT' (if no colon and endswith '/USDT').
        """
        if not symbol:
            return symbol
        s = symbol.strip().upper()
        if ':' in s:
            return s
        if s.endswith('/USDT'):
            return f"{s}:USDT"
        return s

    def _ws_fetch_df_with_fallback(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Try to fetch latest WS OHLCV for timeframe using both raw and normalized symbols.
        Returns a DataFrame if found, otherwise None.
        """
        if not getattr(self, 'websocket_manager', None) or not self.websocket_manager.collector:
            return None

        # Sadece normalize edilmiş sembolü kullan, çünkü collector bu formatta saklıyor.
        sym_norm = self._normalize_symbol_for_ws(symbol)

        try:
            # Doğrudan collector'dan OHLCV listesini al
            ohlcv_list = self.websocket_manager.collector.get_latest_ohlcv(
                exchange='bingx',  # Varsayılan olarak bingx kullanıyoruz
                symbol=sym_norm,
                timeframe=timeframe
            )
            
            if ohlcv_list:
                # Listeyi DataFrame'e çevir
                return self._ohlcv_to_dataframe(ohlcv_list)
        except Exception as e:
            logger.error(f"Error fetching {sym_norm} {timeframe} from collector: {e}")
        
        return None

    def _setup_websocket_connections(self) -> bool:
        """
        Setup WebSocket connections with proper validation and error handling.
        
        Returns:
            True if any streams were started successfully, False otherwise
        """
        # Step 1: Validate Prerequisites
        if not self.exchange_clients:
            logger.error("[WS] ERROR: No exchange clients available")
            return False
        
        # Get symbols with fallback
        fixed_symbols = self.config.get('universe', {}).get('fixed_symbols', [])
        if not fixed_symbols:
            # Fallback to default symbols
            fixed_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
            logger.warning(f"[WS] No symbols configured, using defaults: {fixed_symbols}")
        
        logger.info(f"[WS] Setting up WebSocket streams for {len(fixed_symbols)} symbols")
        
        # Step 2: Initialize Manager
        try:
            if not self.websocket_manager:
                self.websocket_manager = WebSocketManager(exchanges=self.exchange_clients)
                logger.info("[WS] WebSocket manager created")
        except Exception as e:
            logger.error(f"[WS] ERROR: Failed to create WebSocket manager: {type(e).__name__}: {str(e)}")
            return False
        
        # Get timeframes from config
        ws_config = self.config.get('websocket', {})
        timeframes = ws_config.get('stream_timeframes', ['1m', '5m', '15m', '30m', '1h', '4h'])
        
        # Step 3: Setup Streams with Limits
        total_streams_started = 0
        
        for exchange_name in self.exchange_clients.keys():
            # Get stream limit for this exchange
            stream_limit = self._get_stream_limit(exchange_name)
            logger.info(f"[WS] {exchange_name}: stream limit = {stream_limit}")
            
            # Calculate required streams
            required_streams = len(fixed_symbols) * len(timeframes)
            
            # Determine symbols to use (respect limit)
            if required_streams > stream_limit:
                max_symbols = stream_limit // len(timeframes)
                symbols_to_use = fixed_symbols[:max_symbols]
                logger.warning(
                    f"[WS] {exchange_name}: Required {required_streams} streams > limit {stream_limit}, "
                    f"using only first {max_symbols} symbols"
                )
            else:
                symbols_to_use = fixed_symbols
            
            # Start streams for this exchange
            exchange_streams_started = 0
            for symbol in symbols_to_use:
                for tf in timeframes:
                    try:
                        self.websocket_manager.start_ohlcv_stream(exchange_name, symbol, tf)
                        exchange_streams_started += 1
                    except Exception as e:
                        logger.error(f"[WS] ERROR: Failed to start stream {exchange_name}:{symbol}:{tf} - {str(e)}")
            
            total_streams_started += exchange_streams_started
            
            if exchange_streams_started > 0:
                logger.info(f"[WS] {exchange_name}: Started {exchange_streams_started} streams successfully")
                # Update active symbols for this exchange
                if not hasattr(self, 'active_symbols') or not self.active_symbols:
                    self.active_symbols = symbols_to_use
            else:
                logger.error(f"[WS] ERROR: {exchange_name}: No streams started")
        
        # Step 4: Return Status
        if total_streams_started == 0:
            logger.error("[WS] ERROR: No WebSocket streams were started")
            return False
        
        logger.info(f"[WS] ✅ Setup complete: {total_streams_started} total streams started")
        return True
    
    def _get_stream_limit(self, exchange_name: str) -> int:
        """Get stream limit for a specific exchange."""

        # Per-exchange stream limits
        stream_limits = {
            'binance': 50,
            'binanceusdm': 50,
            'bingx': 30,
            'bitget': 30,
            'okx': 40,
        }

        # Try to get from config first
        ws_config = self.config.get('websocket', {})
        max_streams_config = ws_config.get('max_streams_per_exchange', {})
        
        # Return config value, then hardcoded limit, then default
        return max_streams_config.get(
            exchange_name,
            stream_limits.get(exchange_name, 10)  # Default to 10
        )

    def _get_default_symbols(self, limit=20):
        """Get top volume symbols from exchanges."""
        # Basit implementasyon
        default_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT'
        ]
        return default_symbols[:limit]

    async def process_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Process a single symbol for signals.
        This method is now streamlined to use the pre-loaded data from the WebSocket collector.
        """
        try:
            self.processed_symbols_count += 1
            logger.info(f"⚙️ [PROCESS] Processing symbol: {symbol}")

            signals_config = self.config.get('signals', {}) or {}
            mtf_cfg = (signals_config.get('short_the_rip', {}) or {}).get('mtf_confirmation', {}) or {}
            mode_15m_raw = mtf_cfg.get("15m_mode")
            mode_15m = mode_15m_raw.strip().lower() if isinstance(mode_15m_raw, str) else "off"
            mtf_15m_enabled = mode_15m != "off"

            # 1. Veriyi Doğrudan ve Sadece WebSocket Collector'dan Al
            # Bu fonksiyon artık hem geçmiş (primed) hem de canlı veriyi içeren tam DataFrame'i döndürecek.
            df_30m = self._ws_fetch_df_with_fallback(symbol, '30m')
            df_1h = self._ws_fetch_df_with_fallback(symbol, '1h')
            df_4h = self._ws_fetch_df_with_fallback(symbol, '4h')
            
            # Diğer zaman dilimleri stratejiler tarafından direkt kullanılmıyor ama rejim analizi için alınabilir.
            df_1m = self._ws_fetch_df_with_fallback(symbol, '1m')
            df_5m = self._ws_fetch_df_with_fallback(symbol, '5m')
            df_15m = self._ws_fetch_df_with_fallback(symbol, '15m') if mtf_15m_enabled else None

            market_data = {
                '1m': df_1m,
                '5m': df_5m,
                '30m': df_30m,
                '30m_closed': df_30m,
                '30m_hybrid': df_30m,
                '1h': df_1h,
                '4h': df_4h,
            }
            if mtf_15m_enabled:
                market_data['15m'] = df_15m

            # 2. Veri Doğrulama Logları
            if self.debug_logging:
                logger.info(f"📊 [DATA-VALIDATION] For {symbol}:")
                for tf, df in market_data.items():
                    if df is not None and not df.empty:
                        candle_count = len(df)
                        # 200 mum, çoğu indikatör için güvenli bir limittir.
                        status_icon = "✅" if candle_count > 200 else "⚠️"
                        last_close = df['close'].iloc[-1]
                        logger.info(f"  {status_icon} {tf}: {candle_count} candles | Last Close: ${last_close:,.2f}")
                    elif tf in ['30m', '1h']:
                        logger.error(f"  ❌ {tf}: NO DATA AVAILABLE! Strategy cannot run.")
                    else:
                        logger.info(f"  ℹ️ {tf}: No data available for this optional timeframe.")
            
            # Strateji için ana veri (30m) yoksa direkt çık
            if df_30m is None or df_30m.empty:
                logger.warning(f"Skipping {symbol} due to missing primary (30m) data.")
                return None

            # 3. ML Context Generation (NEW)
            ml_context = None
            if hasattr(self, 'ml_integration') and self.ml_integration:
                try:
                    # Prepare market data for ML
                    ml_market_data = {
                        'price_data': df_30m,  # Use 30m as primary for ML
                        'timeframes': market_data
                    }
                    
                    # Get indicator validator if available
                    indicator_validator = None
                    if hasattr(self, 'indicator_validator'):
                        indicator_validator = self.indicator_validator
                    
                    # Get ML context with validation
                    ml_context = await self.ml_integration.get_ml_context(
                        symbol=symbol
                    )
                    
                    # Log ML context status
                    if ml_context.is_healthy:
                        logger.info(
                            f"🧠 [ML] {symbol}: {ml_context.regime_prediction or 'N/A'} "
                            f"(conf={ml_context.regime_confidence:.2%}) | "
                            f"Price: {ml_context.price_direction or 'N/A'} | "
                            f"Consensus: {ml_context.consensus_score:.2%}"
                        )
                    else:
                        logger.warning(
                            f"🧠 [ML] {symbol}: ML context unhealthy - "
                            f"{', '.join(ml_context.validation_errors[:2])}"
                        )
                except Exception as e:
                    logger.warning(f"🧠 [ML] {symbol}: Failed to generate ML context - {e}")
                    # Continue without ML context
                    ml_context = None

            # 4. Market Rejim Analizi (Mevcut haliyle kalabilir)
            metadata = {}
            if self.market_regime_analyzer:
                try:
                    regime = self.market_regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
                    metadata = {'regime': regime}
                except Exception as e:
                    logger.warning(f"[REGIME] Regime analysis failed for {symbol}: {e}")

            # 4. Stratejileri Çalıştır
            logger.info(f"[STRATEGY-CHECK] Executing {len(self.strategies)} strategies for {symbol}")
            for strategy_name, strategy_instance in self.strategies.items():
                logger.info(f"[STRATEGY-CHECK] Running {strategy_name} for {symbol}...")
                try:
                    # Stratejiler artık tüm market verisini ve ML context'i alabilir
                    signal = strategy_instance.signal(
                        df_30m=df_30m, 
                        df_1h=df_1h, 
                        regime_data=metadata.get('regime'), 
                        symbol=symbol,
                        market_data=market_data,
                        ml_context=ml_context  # <<< YENİ: ML Context parametresi
                    )
                    
                    if signal:
                        signal['strategy'] = strategy_name
                        signal['metadata'] = metadata
                        signal['symbol'] = symbol
                        signal['timestamp'] = datetime.now(timezone.utc)
                        entry_indicators = {}
                        if hasattr(df_30m, 'columns') and 'rsi' in df_30m.columns:
                            entry_indicators['rsi'] = float(df_30m['rsi'].iloc[-1])
                        if entry_indicators:
                            signal['entry_indicators'] = entry_indicators
                        
                        # Add ML metadata if available
                        if ml_context and ml_context.is_healthy:
                            signal['ml_metadata'] = {
                                'regime': ml_context.regime_prediction,
                                'regime_confidence': ml_context.regime_confidence,
                                'price_direction': ml_context.price_direction,
                                'consensus': ml_context.consensus_score
                            }
                        
                        logger.info(f"📊 Signal from {strategy_name} for {symbol}: {signal.get('reason')}")
                        return signal # İlk sinyali bul ve döngüden çık
                except Exception as e:
                    logger.error(f"❌ {strategy_name} error for {symbol}: {e}", exc_info=True)
            
            return None # Hiçbir strateji sinyal üretmediyse None dön

        except Exception as e:
            logger.error(f"❌ Critical error processing {symbol}: {e}", exc_info=True)
            return None
            
            # ===== MARKET REGIME ANALYSIS =====
            metadata = {}
            
            if self.market_regime_analyzer:
                try:
                    # Regime analizi yap
                    regime = self.market_regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
                    
                    # Her 10 sembolde bir recommendations logla
                    if self.processed_symbols_count % 10 == 0:
                        recommendations = self.market_regime_analyzer.get_regime_recommendations(df_30m, df_1h, df_4h)
                        if recommendations:
                            logger.info(f"\n📊 MARKET REGIME for {symbol}:")
                            for rec in recommendations[:3]:
                                logger.info(f"  • {rec}")
                    
                    # Strategy uygunluğunu kontrol et
                    ob_favorable, ob_reason = self.market_regime_analyzer.is_favorable_for_strategy(
                        'oversold_bounce', df_30m, df_1h, df_4h
                    )
                    str_favorable, str_reason = self.market_regime_analyzer.is_favorable_for_strategy(
                        'short_the_rip', df_30m, df_1h, df_4h
                    )
                    
                    # Uygun olmayan durumları logla ama atlamak yerine metadata'ya ekle
                    if not ob_favorable and not str_favorable:
                        logger.debug(f"{symbol}: Regime not ideal - OB: {ob_reason}, STR: {str_reason}")
                    
                    # Metadata'ya ekle
                    metadata = {
                        'regime': regime,
                        'ob_favorable': ob_favorable,
                        'str_favorable': str_favorable,
                        'ob_reason': ob_reason,
                        'str_reason': str_reason
                    }
                    
                except Exception as e:
                    logger.warning(f"[REGIME] Regime analysis failed for {symbol}: {e}")
            
            # ===== CREATE & VALIDATE MARKET_DATA DICTIONARY FOR STRATEGIES =====
            market_data = {
                '1m': df_1m,
                '5m': df_5m,
                '30m': df_30m,
                '30m_closed': df_30m,
                '30m_hybrid': df_30m,
                '1h': df_1h,
                '4h': df_4h,
            }
            if mtf_15m_enabled:
                market_data['15m'] = df_15m
            
            # Bu log, verinin gerçekten enjekte edilip edilmediğini kanıtlar.
            # Stratejiye giden verinin kaç mum içerdiğini gösterir.
            if self.debug_logging:
                logger.info(f"📊 [DATA-VALIDATION] For {symbol}:")
                for tf, df in market_data.items():
                    if df is not None and not df.empty:
                        # Enjekte edilmiş verinin kanıtı: Mum sayısının yüksek olması gerekir.
                        candle_count = len(df)
                        status_icon = "✅" if candle_count > 50 else "⚠️" # 50'den fazla mum varsa, enjeksiyon başarılıdır.
                        last_close = df['close'].iloc[-1]
                        logger.info(f"  {status_icon} {tf}: {candle_count} candles | Last Close: ${last_close:,.2f}")
                    elif tf in ['30m', '1h']: # Ana zaman dilimleri eksikse hata ver
                        logger.error(f"  ❌ {tf}: NO DATA AVAILABLE! Strategy cannot run.")
                    else: # Opsiyonel zaman dilimleri eksikse bilgi ver
                        logger.info(f"  ℹ️ {tf}: No data available for this optional timeframe.")
            else:
                 # Debug modu kapalıyken eski, basit logu bas.
                 available_tfs = [tf for tf, df in market_data.items() if df is not None and not df.empty]
                 logger.info(f"[DATA-FETCH] Market data prepared with timeframes: {available_tfs}")
            
            # ===== STRATEGY SIGNALS =====
            signal = None
            
            # Execute registered strategies
            count = len(self.strategies)
            logger.info(f"[STRATEGY-CHECK] {count} registered strategies available")
            
            if count:
                logger.info(f"[STRATEGY-CHECK] Executing {count} strategies for {symbol}")
                
                for strategy_name, strategy_instance in self.strategies.items():
                    logger.info(f"[STRATEGY-CHECK] Running {strategy_name} for {symbol}...")
                    try:
                        # Call strategy's signal method
                        strategy_signal = None
                        
                        # Get cached capabilities
                        capabilities = self.strategy_capabilities.get(strategy_name, {})
                        
                        # Check if strategy has signal method
                        if hasattr(strategy_instance, 'signal'):
                            # Use cached regime_data support check
                            if capabilities.get('supports_regime_data', False):
                                # Adaptive strategies take regime_data parameter and now also market_data
                                strategy_signal = strategy_instance.signal(
                                    df_30m, df_1h, 
                                    regime_data=metadata.get('regime'), 
                                    symbol=symbol,
                                    market_data=market_data
                                )
                            else:
                                # Standard strategies - only pass what they expect
                                strategy_signal = strategy_instance.signal(df_30m, df_1h)
                        elif hasattr(strategy_instance, 'generate_signal'):
                            # Mock or test strategies - use cached async check
                            # Use runtime check to verify if generate_signal is a coroutine function
                            if inspect.iscoroutinefunction(strategy_instance.generate_signal):
                                strategy_signal = await strategy_instance.generate_signal()
                            else:
                                strategy_signal = strategy_instance.generate_signal()
                        
                        if strategy_signal:
                            strategy_signal['strategy'] = strategy_name
                            logger.info(f"📊 Signal from {strategy_name} for {symbol}: {strategy_signal}")
                            signal = strategy_signal
                            break  # Use first signal found
                            
                    except Exception as e:
                        logger.error(f"❌ {strategy_name} error for {symbol}: {e}", exc_info=True)
            else:
                # Fallback: Use default strategies if none registered
                logger.info(f"[STRATEGY-CHECK] No registered strategies, using fallback strategies for {symbol}")
                signals_config = self.config.get('signals', {})
                
                # Check AdaptiveOversoldBounce
                if signals_config.get('oversold_bounce', {}).get('enable', True):
                    logger.info(f"[STRATEGY-CHECK] Checking AdaptiveOversoldBounce (adaptive_ob) for {symbol}")
                    # Sadece regime uygunsa veya ignore_regime true ise
                    ignore_regime = signals_config.get('oversold_bounce', {}).get('ignore_regime', False)
                    
                    if metadata.get('ob_favorable', True) or ignore_regime:
                        try:
                            ob_config = signals_config.get('oversold_bounce', {})
                            ob = AdaptiveOversoldBounce(ob_config, self.market_regime_analyzer)
                            
                            # Adaptive strateji farklı parametre alıyor - market_data ekle
                            signal = ob.signal(
                                df_30m, df_1h, 
                                regime_data=metadata.get('regime'), 
                                symbol=symbol,
                                market_data=market_data
                            )
                            
                            if signal:
                                signal['strategy'] = 'adaptive_ob'
                                logger.info(f"📊 Signal from adaptive_ob for {symbol}: {signal}")
                            else:
                                logger.info(f"[STRATEGY-CHECK] adaptive_ob: No signal for {symbol}")
                                
                        except Exception as e:
                            logger.warning(f"[STRATEGY-CHECK] AdaptiveOB error for {symbol}: {e}", exc_info=True)
                    else:
                        logger.info(f"[STRATEGY-CHECK] adaptive_ob: Regime not favorable for {symbol}, skipping")
                
                # ---------------------------------------------------------
                # Check AdaptiveShortTheRip (sadece signal yoksa)
                if not signal and signals_config.get('short_the_rip', {}).get('enable', True):
                    logger.info(f"[STRATEGY-CHECK] Checking AdaptiveShortTheRip (adaptive_str) for {symbol}")
                    ignore_regime = signals_config.get('short_the_rip', {}).get('ignore_regime', False)
                    
                    if metadata.get('str_favorable', True) or ignore_regime:
                        try:
                            str_config = signals_config.get('short_the_rip', {})
                            strp = AdaptiveShortTheRip(str_config, self.market_regime_analyzer)
                            
                            # market_data parametresi ekle
                            signal = strp.signal(
                                df_30m, df_1h, 
                                regime_data=metadata.get('regime'), 
                                symbol=symbol,
                                market_data=market_data
                            )
                            
                            if signal:
                                signal['strategy'] = 'adaptive_str'
                                logger.info(f"📊 Signal from adaptive_str for {symbol}: {signal}")
                            else:
                                logger.info(f"[STRATEGY-CHECK] adaptive_str: No signal for {symbol}")
                                
                        except Exception as e:
                            logger.warning(f"[STRATEGY-CHECK] AdaptiveSTR error for {symbol}: {e}", exc_info=True)
                    else:
                        logger.info(f"[STRATEGY-CHECK] adaptive_str: Regime not favorable for {symbol}, skipping")
            
            # Signal'e metadata ekle
            if signal:
                signal['metadata'] = metadata
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now(timezone.utc)

            # Monitor adaptive signals
            if signal and signal.get('is_adaptive'):
                from core.adaptive_monitor import adaptive_monitor
                adaptive_monitor.record_adaptive_signal(symbol, signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Critical error processing {symbol}: {e}", exc_info=True)
            return None 

    async def _process_trading_loop(self):
        """The core processing logic for a single trading loop iteration."""
        if not self.active_symbols:
            logger.warning("[PROCESS] No active symbols to process.")
            return

        logger.info(f"📋 [PROCESSING] Starting processing loop for {len(self.active_symbols)} symbols")
        start_time = time.time()
        
        if not self.portfolio_manager or not hasattr(self.portfolio_manager, 'strategies'):
             logger.error("❌ PortfolioManager or strategies not initialized. Cannot process loop.")
             return
        strategies_to_run = list(self.portfolio_manager.strategies.items())
        hybrid_allowlist = {"adaptive_ob", "adaptive_str"}
        hybrid_strategies = {
            name
            for name, instance in strategies_to_run
            if (name in hybrid_allowlist) or (getattr(instance, "strategy_name", "") in hybrid_allowlist)
        }

        signals_config = self.config.get('signals', {}) or {}
        mtf_cfg = (signals_config.get('short_the_rip', {}) or {}).get('mtf_confirmation', {}) or {}
        mode_15m_raw = mtf_cfg.get("15m_mode")
        mode_15m = mode_15m_raw.strip().lower() if isinstance(mode_15m_raw, str) else "off"
        mtf_15m_enabled = mode_15m != "off"
        mode_1h_raw = mtf_cfg.get("1h_mode")
        mode_1h = mode_1h_raw.strip().lower() if isinstance(mode_1h_raw, str) else "off"
        mtf_any_enabled = mtf_15m_enabled or mode_1h != "off"

        processed_count = 0
        error_count = 0

        for symbol in self.active_symbols:
            logger.info(f"⚙️ [PROCESS] Processing symbol: {symbol}")
            
            # --- DATA FETCHING STAGE ---
            ml_context = None
            df_30m = None
            df_30m_closed = None
            df_30m_hybrid = None
            df_1h = None
            df_15m = None
            
            try:
                # 1. Get ML Context first
                if self.ml_integration:
                    # This now correctly calls get_ml_context from the integration manager
                    ml_context = await self.ml_integration.get_ml_context(symbol)
                    if not ml_context or not ml_context.get('is_healthy'):
                        reason = ml_context.get('reason', 'unknown') if ml_context else 'unknown'
                        logger.warning(f"🧠 [ML] {symbol}: ML context is unhealthy. Reason: {reason}")
                
                # --- PPO MONITORING (Shadow Mode) ---
                # Force PPO inference for telemetry even if no signal is generated
                if self.strategy_coordinator:
                    await self.strategy_coordinator.monitor_ppo_state(symbol)

                # 2. Get indicator data directly from MarketDataPipeline
                if self.market_data_pipeline:
                    df_30m_closed = await self.market_data_pipeline.get_latest_ohlcv(
                        symbol, "30m", include_forming=False
                    )
                    if hybrid_strategies:
                        df_30m_hybrid = await self.market_data_pipeline.get_latest_ohlcv(
                            symbol, "30m", include_forming=True
                        )
                    # Default view for downstream logging
                    df_30m = (
                        df_30m_hybrid
                        if df_30m_hybrid is not None and not df_30m_hybrid.empty
                        else df_30m_closed
                    )
                    if df_30m is None or df_30m.empty:
                        raise RuntimeError("30m OHLCV unavailable: both hybrid and closed are empty/None")
                    df_1h = await self.market_data_pipeline.get_latest_ohlcv(symbol, "1h")
                    if mtf_15m_enabled:
                        df_15m = await self.market_data_pipeline.get_latest_ohlcv(symbol, "15m")
                else:
                    logger.error("❌ MarketDataPipeline is not available in ProductionCoordinator.")
                    error_count += 1
                    continue

                if df_30m is None or df_30m.empty:
                    logger.warning(f"⚠️ Could not retrieve 30m indicator data for {symbol}. Skipping symbol.")
                    error_count += 1
                    continue
                
                if df_1h is None or df_1h.empty:
                    logger.warning(f"⚠️ Could not retrieve 1h indicator data for {symbol}, but proceeding with 30m data only.")
            
            except Exception as data_fetch_error:
                logger.error(f"❌ Critical error during data fetching for {symbol}: {data_fetch_error}", exc_info=True)
                error_count += 1
                continue

            processed_count += 1
            
            # --- STRATEGY EXECUTION AND SIGNAL FORWARDING STAGE ---
            market_data = {
                '30m': df_30m,
                '30m_closed': df_30m_closed,
                '30m_hybrid': df_30m_hybrid,
                '1h': df_1h,
            }
            if mtf_15m_enabled:
                market_data['15m'] = df_15m

            for strategy_name, strategy_instance in strategies_to_run:
                try:
                    if not self.portfolio_manager.strategy_metadata.get(strategy_name, {}).get('active', False):
                        logger.debug(f"Skipping inactive strategy: {strategy_name}")
                        continue
                    
                    if hasattr(strategy_instance, 'signal') and callable(getattr(strategy_instance, 'signal')):
                        # ✅ FIX: Pass the fresh ml_context directly to the strategy's 'signal' method
                        strategy_df_30m = df_30m_closed
                        if hybrid_strategies and (strategy_name in hybrid_strategies) and df_30m_hybrid is not None:
                            strategy_df_30m = df_30m_hybrid

                        signal_kwargs = {
                            'df_30m': strategy_df_30m,
                            'df_1h': df_1h,
                            'regime_data': ml_context,  # Pass the entire context
                            'symbol': symbol,
                            'ml_context': ml_context,  # Pass it again for explicit clarity
                        }
                        if market_data is not None:
                            try:
                                if 'market_data' in inspect.signature(strategy_instance.signal).parameters:
                                    strategy_market_data = dict(market_data)
                                    strategy_market_data['30m'] = strategy_df_30m
                                    signal_kwargs['market_data'] = strategy_market_data
                            except (TypeError, ValueError):
                                pass
                        result = strategy_instance.signal(**signal_kwargs)
                        if inspect.iscoroutine(result):
                            signal = await result
                        else:
                            signal = result
                        
                        if signal:
                            logger.info(f"💡 Signal generated by '{strategy_name}' for {symbol}. Forwarding to StrategyCoordinator.")
                            await self._route_strategy_output(strategy_name, signal)
                    else:
                        logger.error(f"❌ Strategy '{strategy_name}' does not have a callable 'signal' method.")
                        error_count += 1

                except Exception as e:
                    logger.error(f"❌ Error running strategy '{strategy_name}' for {symbol}: {e}", exc_info=True)
                    error_count += 1

        await self._nudge_strategy_dispatch()
        if self.strategy_coordinator and hasattr(self.strategy_coordinator, "incubator_tick"):
            try:
                await self.strategy_coordinator.incubator_tick()
            except Exception as exc:
                logger.debug("[PROCESS] incubator_tick failed: %s", exc)

        try:
            await self._drain_strategy_recheck_requests()
        except Exception as exc:
            logger.debug("[PROCESS] strategy_recheck_request drain failed: %s", exc)

        end_time = time.time()
        logger.info(f"✅ [PROCESSING] Completed processing loop in {end_time - start_time:.2f}s")
        logger.info(f"   Processed: {processed_count}/{len(self.active_symbols)} symbols | Errors: {error_count}")

    async def _nudge_strategy_dispatch(self) -> None:
        """Drain the coordinator queue once per loop to keep execution responsive."""
        if not self.strategy_coordinator:
            return

        # Prefer LiveTradingEngine helper so signals follow the normal bridge path
        if self.trading_engine and hasattr(self.trading_engine, 'trigger_coordinator_drain'):
            try:
                drained = await self.trading_engine.trigger_coordinator_drain(timeout=0.0)
            except Exception as exc:
                logger.debug(f"[PROCESS] Queue drain via trading engine failed: {exc}")
                drained = False
            if drained:
                return

        dispatcher = getattr(self.strategy_coordinator, 'try_dispatch_next', None)
        if not callable(dispatcher):
            return

        try:
            payload = await dispatcher(timeout=0.0)
        except Exception as exc:
            logger.debug(f"[PROCESS] Strategy queue dispatch failed: {exc}")
            return

        if not payload:
            return

        if self.trading_engine and hasattr(self.trading_engine, '_forward_signal_from_coordinator'):
            await self.trading_engine._forward_signal_from_coordinator(payload)
            return

        queue = getattr(self.strategy_coordinator, 'signal_queue', None)
        if queue and hasattr(queue, 'requeue'):
            await queue.requeue(payload)

    async def _route_strategy_output(self, strategy_name: str, payload: Any) -> Optional[Dict[str, Any]]:
        coordinator = getattr(self, "strategy_coordinator", None)
        if not coordinator or not payload:
            return None

        if isinstance(payload, dict) and payload.get("event_type") == "soft_deferral_event":
            event = dict(payload)
            event.setdefault("strategy", strategy_name)
            handler = getattr(coordinator, "handle_soft_deferral", None)
            if not callable(handler):
                return {"status": "rejected", "reason": "soft_deferral_handler_unavailable", "stage": "soft_deferral"}
            return await handler(event)

        return await coordinator.process_strategy_signal(strategy_name, payload)

    async def _drain_strategy_recheck_requests(self) -> None:
        """Consume strategy_recheck_request events emitted by StrategyCoordinator."""
        coordinator = getattr(self, "strategy_coordinator", None)
        if not coordinator:
            return

        queue = getattr(coordinator, "strategy_recheck_queue", None)
        if queue is None or not hasattr(queue, "get_nowait"):
            return

        cfg = getattr(self, "config", {}) or {}
        max_rechecks_per_loop = 10
        try:
            raw = ((cfg.get("incubator") or {}).get("max_rechecks_per_loop"))
            if raw is None:
                raw = ((cfg.get("soft_deferral") or {}).get("max_rechecks_per_loop"))
            if raw is not None:
                max_rechecks_per_loop = int(raw)
        except Exception:
            max_rechecks_per_loop = 10

        max_rechecks_per_loop = max(0, int(max_rechecks_per_loop))
        if max_rechecks_per_loop <= 0:
            return
        recheck_dedupe_mode = str(((cfg.get("incubator") or {}).get("recheck_dedupe_mode")) or "strategy_symbol_side").strip().lower()
        if recheck_dedupe_mode not in ("strategy_symbol_side", "strategy_symbol_side_timeframe"):
            recheck_dedupe_mode = "strategy_symbol_side"

        scan_limit = min(1000, max_rechecks_per_loop * 50)
        scan_limit = max(max_rechecks_per_loop, scan_limit)

        drained: List[Any] = []
        for _ in range(scan_limit):
            try:
                drained.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            except Exception:
                break

        if not drained:
            return

        def _emit_dropped_recheck(ev: Dict[str, Any], *, dropped_reason: str) -> None:
            try:
                run_id = get_current_run_id()
            except Exception:
                run_id = None

            strategy = ev.get("strategy") or ev.get("strategy_name")
            symbol = ev.get("symbol")
            parent_pending_id = ev.get("parent_pending_id") or ev.get("pending_id")
            if parent_pending_id is not None:
                try:
                    parent_pending_id = str(parent_pending_id)
                except Exception:
                    parent_pending_id = None

            side_raw = ev.get("side")
            canonical_side = None
            if side_raw is not None:
                try:
                    side_raw = str(side_raw).strip().lower()
                except Exception:
                    side_raw = ""
                canonical_side = {"buy": "long", "long": "long", "sell": "short", "short": "short"}.get(side_raw)

            timeframe_raw = ev.get("timeframe") or ev.get("tf")
            timeframe = None
            if timeframe_raw is not None:
                try:
                    timeframe = str(timeframe_raw).strip().lower()
                except Exception:
                    timeframe = None

            attempt = None
            if parent_pending_id:
                tracker = getattr(self, "_soft_deferral_recheck_attempts", None)
                if isinstance(tracker, dict):
                    try:
                        attempt = int(tracker.get(parent_pending_id, 0) or 0) + 1
                    except Exception:
                        attempt = 1
                else:
                    attempt = 1

            pending_reason_code = ev.get("pending_reason_code") or ev.get("reason_code")
            if pending_reason_code is not None:
                try:
                    pending_reason_code = str(pending_reason_code)
                except Exception:
                    pending_reason_code = None

            out = {
                "event": "soft_deferral_recheck_outcome",
                "ts_ms": int(time.time() * 1000),
                "run_id": run_id,
                "parent_pending_id": parent_pending_id,
                "strategy": str(strategy) if strategy is not None else None,
                "symbol": str(symbol) if symbol is not None else None,
                "side": canonical_side,
                "timeframe": timeframe,
                "attempt": attempt,
                "pending_reason_code": pending_reason_code,
                "outcome": "dropped_deduped",
                "dropped_reason": str(dropped_reason),
                "elapsed_ms": 0,
            }
            try:
                logger.info(
                    "soft_deferral_recheck_outcome %s",
                    json.dumps(out, ensure_ascii=False, sort_keys=True),
                )
            except Exception:
                logger.info("soft_deferral_recheck_outcome %s", out)

        dropped: List[Tuple[Dict[str, Any], str]] = []
        latest_by_key: Dict[str, Dict[str, Any]] = {}
        for event in drained:
            if not isinstance(event, dict):
                continue

            strategy = event.get("strategy") or event.get("strategy_name")
            symbol = event.get("symbol")
            if not strategy or not symbol:
                continue

            side_raw = event.get("side")
            side = None
            if side_raw is not None:
                try:
                    side_raw = str(side_raw).strip().lower()
                except Exception:
                    side_raw = ""
                side = {"buy": "long", "long": "long", "sell": "short", "short": "short"}.get(side_raw)
            timeframe_raw = event.get("timeframe") or event.get("tf")
            timeframe = None
            if timeframe_raw is not None:
                try:
                    timeframe = str(timeframe_raw).strip().lower()
                except Exception:
                    timeframe = None
            if recheck_dedupe_mode == "strategy_symbol_side_timeframe":
                key = f"{strategy}:{symbol}:{side or 'unknown'}:{timeframe or 'unknown'}"
            else:
                key = f"{strategy}:{symbol}:{side or 'unknown'}"
            try:
                ts_ms = int(event.get("ts_ms") or 0)
            except Exception:
                ts_ms = 0

            existing = latest_by_key.get(key)
            if existing is None:
                latest_by_key[key] = event
                continue

            try:
                existing_ts_ms = int(existing.get("ts_ms") or 0)
            except Exception:
                existing_ts_ms = 0

            if ts_ms >= existing_ts_ms:
                dropped.append((existing, "deduped_older_ts"))
                latest_by_key[key] = event
            else:
                dropped.append((event, "deduped_older_ts"))

        deduped = list(latest_by_key.values())
        if not deduped:
            return

        def _event_ts_ms(ev: Dict[str, Any]) -> int:
            try:
                return int(ev.get("ts_ms") or 0)
            except Exception:
                return 0

        deduped.sort(key=_event_ts_ms, reverse=True)
        to_process = deduped[:max_rechecks_per_loop]
        to_requeue = deduped[max_rechecks_per_loop:]

        for event in to_requeue:
            if isinstance(event, dict):
                dropped.append((event, "over_capacity"))

        for ev, dropped_reason in dropped:
            if isinstance(ev, dict):
                _emit_dropped_recheck(ev, dropped_reason=dropped_reason)

        for event in to_requeue:
            try:
                queue.put_nowait(event)
            except Exception:
                break

        for event in to_process:
            try:
                await self._handle_strategy_recheck_request(event)
            except Exception as exc:
                logger.debug("[SOFT-DEFERRAL] strategy_recheck_request handling failed: %s", exc, exc_info=True)

    def notify_recheck_ready(self) -> None:
        event = getattr(self, "_recheck_wakeup_event", None)
        if event is not None:
            try:
                event.set()
            except Exception:
                pass

    async def _sleep_with_recheck_wakeup(self, sleep_time: float) -> None:
        if sleep_time <= 0:
            return
        event = getattr(self, "_recheck_wakeup_event", None)
        if event is None:
            await asyncio.sleep(sleep_time)
            return
        deadline = time.monotonic() + float(sleep_time)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except Exception:
                break
            try:
                event.clear()
            except Exception:
                pass
            try:
                await self._drain_strategy_recheck_requests()
            except Exception as exc:
                logger.debug("[PROCESS] strategy_recheck_request drain failed: %s", exc)

    async def _handle_strategy_recheck_request(self, event: Any) -> bool:
        if not isinstance(event, dict):
            logger.debug("[SOFT-DEFERRAL] Ignoring invalid strategy_recheck_request: %s", event)
            return False

        strategy = event.get("strategy") or event.get("strategy_name")
        symbol = event.get("symbol")
        if not strategy or not symbol:
            logger.debug("[SOFT-DEFERRAL] Ignoring incomplete strategy_recheck_request: %s", event)
            return False
        parent_pending_id = event.get("parent_pending_id") or event.get("pending_id")
        if parent_pending_id is not None:
            try:
                parent_pending_id = str(parent_pending_id)
            except Exception:
                parent_pending_id = None
        pending_reason_code = event.get("pending_reason_code") or event.get("reason_code")
        if pending_reason_code is not None:
            try:
                pending_reason_code = str(pending_reason_code)
            except Exception:
                pending_reason_code = None
        pending_id = event.get("pending_id")
        if pending_id is not None:
            try:
                pending_id = str(pending_id)
            except Exception:
                pending_id = None
        side = event.get("side")
        timeframe = event.get("timeframe") or event.get("tf")
        condition_data = event.get("condition_data") if isinstance(event.get("condition_data"), dict) else None
        check_detail = event.get("check_detail") if isinstance(event.get("check_detail"), dict) else None
        refresh_policy = event.get("refresh_policy")

        dispatcher = getattr(self, "dispatch_strategy", None)
        if not callable(dispatcher):
            logger.error("[SOFT-DEFERRAL] dispatch_strategy not available; dropping recheck request")
            return False

        result = dispatcher(
            str(symbol),
            str(strategy),
            parent_pending_id=parent_pending_id,
            pending_id=pending_id,
            side=side,
            timeframe=timeframe,
            pending_reason_code=pending_reason_code,
            condition_data=condition_data,
            check_detail=check_detail,
            return_detail=True,
        )
        detail = await result if inspect.iscoroutine(result) else result
        if isinstance(detail, dict):
            dispatched = bool(detail.get("dispatched"))
            rearm_fast_watch = bool(detail.get("rearm_fast_watch"))
            decision_meta = detail.get("decision_meta") if isinstance(detail.get("decision_meta"), dict) else {}
            interval_hint_ms = detail.get("rearm_interval_ms")
            final_reason = detail.get("final_reason")
        else:
            dispatched = bool(detail)
            rearm_fast_watch = False
            decision_meta = {}
            interval_hint_ms = None
            final_reason = None

        if dispatched:
            logger.info("Soft Deferral Recheck Dispatched: %s %s", str(strategy), str(symbol))

        coordinator = getattr(self, "strategy_coordinator", None)
        if (
            coordinator
            and str(refresh_policy or "").upper() == "FAST_PRICE_WATCH"
            and hasattr(coordinator, "on_recheck_result")
            and pending_id
        ):
            rearm_result = await coordinator.on_recheck_result(
                pending_id,
                rearm=bool(rearm_fast_watch),
                interval_hint_ms=interval_hint_ms,
                decision_meta=decision_meta if isinstance(decision_meta, dict) else None,
                final_reason=final_reason,
            )
            if rearm_fast_watch:
                out = {
                    "event": "soft_deferral_rearm",
                    "ts_ms": int(time.time() * 1000),
                    "run_id": get_current_run_id(),
                    "pending_id": pending_id,
                    "parent_pending_id": parent_pending_id,
                    "strategy": str(strategy),
                    "symbol": str(symbol),
                    "side": side,
                    "timeframe": timeframe,
                    "near": decision_meta.get("near"),
                    "dist_to_trigger_bps": decision_meta.get("dist_to_trigger_bps"),
                    "eps_bps": decision_meta.get("eps_bps"),
                    "rearm_count": rearm_result.get("rearm_count") if isinstance(rearm_result, dict) else None,
                    "remaining_ttl_ms": rearm_result.get("remaining_ttl_ms") if isinstance(rearm_result, dict) else None,
                    "next_interval_ms": rearm_result.get("next_interval_ms") if isinstance(rearm_result, dict) else None,
                }
                safe_out = coordinator._json_sanitize(out) if hasattr(coordinator, "_json_sanitize") else out
                logger.info("soft_deferral_rearm %s", json.dumps(safe_out, ensure_ascii=False, sort_keys=True))

        return dispatched

    async def dispatch_strategy(
        self,
        symbol: str,
        strategy: str,
        *,
        parent_pending_id: Optional[str] = None,
        pending_id: Optional[str] = None,
        side: Optional[str] = None,
        timeframe: Optional[str] = None,
        pending_reason_code: Optional[str] = None,
        condition_data: Optional[Dict[str, Any]] = None,
        check_detail: Optional[Dict[str, Any]] = None,
        return_detail: bool = False,
    ) -> bool:
        """Trigger a targeted strategy evaluation for a single symbol."""
        start_monotonic = time.monotonic()
        run_id = None
        try:
            run_id = get_current_run_id()
        except Exception:
            run_id = None

        parent_pending_id_str = None
        if parent_pending_id is not None:
            try:
                parent_pending_id_str = str(parent_pending_id)
            except Exception:
                parent_pending_id_str = None

        canonical_side = None
        if side is not None:
            try:
                side_raw = str(side).strip().lower()
            except Exception:
                side_raw = ""
            canonical_side = {"buy": "long", "long": "long", "sell": "short", "short": "short"}.get(side_raw)

        timeframe_str = None
        if timeframe is not None:
            try:
                timeframe_str = str(timeframe).strip().lower()
            except Exception:
                timeframe_str = None

        attempt = None
        if parent_pending_id_str:
            tracker = getattr(self, "_soft_deferral_recheck_attempts", None)
            if not isinstance(tracker, dict):
                tracker = {}
                setattr(self, "_soft_deferral_recheck_attempts", tracker)
            try:
                attempt = int(tracker.get(parent_pending_id_str, 0) or 0) + 1
            except Exception:
                attempt = 1
            tracker[parent_pending_id_str] = attempt

        outcome = "error"
        emitted_signal_id = None
        error_code = None
        error_detail = None
        limit_requested_by_tf: Dict[str, int] = {}
        rows_returned_by_tf: Dict[str, int] = {}
        strategy_recheck_debug: Optional[Dict[str, Any]] = None
        rearm_fast_watch = False
        decision_meta: Dict[str, Any] = {}
        rearm_interval_ms = None
        final_reason = None
        try:
            if not symbol or not strategy:
                error_code = "invalid_args"
                return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False

            if not getattr(self, "portfolio_manager", None) or not hasattr(self.portfolio_manager, "strategies"):
                error_code = "missing_portfolio_manager"
                return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False

            if not getattr(self, "market_data_pipeline", None):
                error_code = "missing_market_data_pipeline"
                return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False

            if not getattr(self, "strategy_coordinator", None):
                error_code = "missing_strategy_coordinator"
                return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False

            strategies = getattr(self.portfolio_manager, "strategies", {}) or {}
            strategy_instance = strategies.get(strategy)
            if strategy_instance is None:
                for name, instance in strategies.items():
                    if getattr(instance, "strategy_name", None) == strategy:
                        strategy_instance = instance
                        strategy = name
                        break

            if strategy_instance is None or not hasattr(strategy_instance, "signal") or not callable(getattr(strategy_instance, "signal")):
                error_code = "strategy_unavailable"
                return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False

            try:
                if hasattr(self.portfolio_manager, "strategy_metadata"):
                    meta = self.portfolio_manager.strategy_metadata.get(strategy, {}) or {}
                    if not meta.get("active", True):
                        error_code = "strategy_inactive"
                        return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False
            except Exception:
                pass

            ml_context = None
            ml_integration = getattr(self, "ml_integration", None)
            if ml_integration and hasattr(ml_integration, "get_ml_context"):
                try:
                    ml_context = await ml_integration.get_ml_context(symbol)
                except Exception:
                    ml_context = None

            is_mean_reversion = (getattr(strategy_instance, "strategy_name", None) == "mean_reversion") or (
                str(strategy).strip().lower() == "mean_reversion"
            )

            signal_kwargs: Dict[str, Any] = {
                "symbol": symbol,
                "ml_context": ml_context,
                "regime_data": ml_context,
            }
            if parent_pending_id_str:
                signal_kwargs["parent_pending_id"] = parent_pending_id_str
            if pending_reason_code:
                signal_kwargs["pending_reason_code"] = str(pending_reason_code)
            if is_mean_reversion and parent_pending_id_str:
                if pending_id:
                    signal_kwargs["pending_id"] = pending_id
                if condition_data is not None:
                    signal_kwargs["condition_data"] = condition_data
                if check_detail is not None:
                    signal_kwargs["check_detail"] = check_detail

            df_30m_closed = None
            df_30m_hybrid = None
            df_1h = None
            if not is_mean_reversion:
                limit_30m = 1000 if parent_pending_id_str else None
                limit_1h = 1000 if parent_pending_id_str else None
                if limit_30m is not None:
                    limit_requested_by_tf["30m"] = int(limit_30m)
                if limit_1h is not None:
                    limit_requested_by_tf["1h"] = int(limit_1h)

                df_30m_closed = await self.market_data_pipeline.get_latest_ohlcv(
                    symbol,
                    "30m",
                    limit=limit_30m,
                    include_forming=False,
                )
                if df_30m_closed is None or getattr(df_30m_closed, "empty", False):
                    error_code = "missing_ohlcv_30m"
                    return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False
                try:
                    rows_returned_by_tf["30m"] = int(len(df_30m_closed))
                except Exception:
                    rows_returned_by_tf["30m"] = 0

                try:
                    if strategy in {"adaptive_ob", "adaptive_str"}:
                        df_30m_hybrid = await self.market_data_pipeline.get_latest_ohlcv(
                            symbol,
                            "30m",
                            limit=limit_30m,
                            include_forming=True,
                        )
                except Exception:
                    df_30m_hybrid = None

                df_1h = await self.market_data_pipeline.get_latest_ohlcv(symbol, "1h", limit=limit_1h)
                if df_1h is None or getattr(df_1h, "empty", False):
                    error_code = "missing_ohlcv_1h"
                    return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False
                try:
                    rows_returned_by_tf["1h"] = int(len(df_1h))
                except Exception:
                    rows_returned_by_tf["1h"] = 0

                strategy_df_30m = df_30m_closed
                if df_30m_hybrid is not None and not getattr(df_30m_hybrid, "empty", True):
                    strategy_df_30m = df_30m_hybrid

                signal_kwargs.update(
                    {
                        "df_30m": strategy_df_30m,
                        "df_1h": df_1h,
                    }
                )
            else:
                vwap_tf = getattr(strategy_instance, "vwap_tf", None) or "1m"
                signal_tf = getattr(strategy_instance, "signal_tf", None) or "5m"
                try:
                    vwap_limit = int(getattr(strategy_instance, "min_rows", 0) or 0) or None
                except Exception:
                    vwap_limit = None
                try:
                    sig_limit = int(getattr(strategy_instance, "min_signal_rows", 0) or 0) or None
                except Exception:
                    sig_limit = None

                # STRATEGY_RECHECK guardrail: never undercut a safe baseline with tiny explicit limits.
                # For rechecks we default to >= 1000 bars unless the strategy requires more.
                if parent_pending_id_str:
                    try:
                        vwap_limit_int = int(vwap_limit or 0)
                    except Exception:
                        vwap_limit_int = 0
                    try:
                        sig_limit_int = int(sig_limit or 0)
                    except Exception:
                        sig_limit_int = 0
                    vwap_limit = max(vwap_limit_int, 1000)
                    sig_limit = max(sig_limit_int, 1000)

                vwap_tf_str = str(vwap_tf)
                signal_tf_str = str(signal_tf)
                if vwap_limit is not None:
                    limit_requested_by_tf[vwap_tf_str] = int(vwap_limit)
                if sig_limit is not None:
                    limit_requested_by_tf[signal_tf_str] = int(sig_limit)

                df_vwap = await self.market_data_pipeline.get_latest_ohlcv(symbol, vwap_tf_str, limit=vwap_limit)
                df_sig = await self.market_data_pipeline.get_latest_ohlcv(symbol, signal_tf_str, limit=sig_limit)
                if df_vwap is None or getattr(df_vwap, "empty", False) or df_sig is None or getattr(df_sig, "empty", False):
                    error_code = "missing_ohlcv_mean_reversion"
                    return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False
                try:
                    rows_returned_by_tf[vwap_tf_str] = int(len(df_vwap))
                except Exception:
                    rows_returned_by_tf[vwap_tf_str] = 0
                try:
                    rows_returned_by_tf[signal_tf_str] = int(len(df_sig))
                except Exception:
                    rows_returned_by_tf[signal_tf_str] = 0

                signal_kwargs.update({"df_vwap": df_vwap, "df_sig": df_sig})

            try:
                if "market_data" in inspect.signature(strategy_instance.signal).parameters:
                    market_data: Dict[str, Any] = {}
                    if is_mean_reversion:
                        market_data.update(
                            {
                                "df_vwap": signal_kwargs.get("df_vwap"),
                                "df_sig": signal_kwargs.get("df_sig"),
                                str(getattr(strategy_instance, "vwap_tf", "1m")): signal_kwargs.get("df_vwap"),
                                str(getattr(strategy_instance, "signal_tf", "5m")): signal_kwargs.get("df_sig"),
                                symbol: {
                                    str(getattr(strategy_instance, "vwap_tf", "1m")): signal_kwargs.get("df_vwap"),
                                    str(getattr(strategy_instance, "signal_tf", "5m")): signal_kwargs.get("df_sig"),
                                },
                            }
                        )
                    else:
                        market_data.update(
                            {
                                "30m": signal_kwargs.get("df_30m"),
                                "30m_closed": df_30m_closed,
                                "30m_hybrid": df_30m_hybrid,
                                "1h": df_1h,
                            }
                        )
                    signal_kwargs["market_data"] = market_data
            except Exception:
                pass

            raw = strategy_instance.signal(**signal_kwargs)
            signal = await raw if inspect.iscoroutine(raw) else raw
            if isinstance(signal, dict) and signal.get("event_type") == "strategy_recheck_decision":
                decision_meta = signal.get("decision_meta") if isinstance(signal.get("decision_meta"), dict) else {}
                rearm_fast_watch = bool(decision_meta.get("rearm_fast_watch"))
                rearm_interval_ms = decision_meta.get("rearm_backoff_ms")
                outcome = "rearmed" if rearm_fast_watch else "no_signal"
                final_reason = decision_meta.get("rearm_reason") or "recheck_hold"
                return {"dispatched": False, "rearm_fast_watch": rearm_fast_watch, "decision_meta": decision_meta, "rearm_interval_ms": rearm_interval_ms, "final_reason": final_reason} if return_detail else False
            if not signal:
                outcome = "no_signal"
                final_reason = "recheck_hold"
                logger.info(
                    "Soft Deferral Recheck yielded NO SIGNAL for pending_id=%s strategy=%s symbol=%s",
                    str(parent_pending_id_str),
                    str(strategy),
                    str(symbol),
                )
                return {"dispatched": False, "rearm_fast_watch": False, "decision_meta": {}, "final_reason": final_reason} if return_detail else False

            if isinstance(signal, dict):
                signal.setdefault("symbol", symbol)
                signal.setdefault("timestamp", datetime.now(timezone.utc))
                if parent_pending_id_str:
                    try:
                        meta = signal.get("meta")
                        if not isinstance(meta, dict):
                            meta = {}
                            signal["meta"] = meta
                        meta.setdefault("parent_pending_id", parent_pending_id_str)
                        if pending_reason_code:
                            meta.setdefault("pending_reason_code", str(pending_reason_code))
                            signal.setdefault("pending_reason_code", str(pending_reason_code))
                    except Exception:
                        pass

                    try:
                        meta = signal.get("meta")
                        if isinstance(meta, dict) and isinstance(meta.get("recheck_debug"), dict):
                            strategy_recheck_debug = dict(meta.get("recheck_debug") or {})
                    except Exception:
                        strategy_recheck_debug = None

                if parent_pending_id_str and signal.get("event_type") == "soft_deferral_event":
                    outcome = "error"
                    error_code = "loop_prevented"
                    error_detail = "soft_deferral_returned_on_recheck"
                    return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False

            route_result = await self._route_strategy_output(strategy, signal)
            if isinstance(route_result, dict) and route_result.get("signal_id"):
                try:
                    emitted_signal_id = str(route_result.get("signal_id"))
                except Exception:
                    emitted_signal_id = route_result.get("signal_id")
            elif isinstance(signal, dict) and signal.get("signal_id"):
                try:
                    emitted_signal_id = str(signal.get("signal_id"))
                except Exception:
                    emitted_signal_id = signal.get("signal_id")
            outcome = "signal_emitted"
            final_reason = "signal_emitted"
            return {"dispatched": True, "rearm_fast_watch": False, "decision_meta": {}, "final_reason": final_reason} if return_detail else True
        except Exception as exc:
            outcome = "error"
            error_code = "exception"
            error_detail = str(exc)
            logger.debug("[SOFT-DEFERRAL] dispatch_strategy failed: %s", exc, exc_info=True)
            return {"dispatched": False, "rearm_fast_watch": False} if return_detail else False
        finally:
            if parent_pending_id_str:
                elapsed_ms = int((time.monotonic() - start_monotonic) * 1000)
                if outcome == "error" and not error_detail and error_code:
                    error_detail = str(error_code)
                event_out: Dict[str, Any] = {
                    "event": "soft_deferral_recheck_outcome",
                    "ts_ms": int(time.time() * 1000),
                    "run_id": run_id,
                    "parent_pending_id": parent_pending_id_str,
                    "strategy": str(strategy),
                    "symbol": str(symbol),
                    "side": canonical_side,
                    "timeframe": timeframe_str,
                    "attempt": attempt,
                    "pending_reason_code": pending_reason_code,
                    "outcome": outcome,
                    "elapsed_ms": elapsed_ms,
                    "limit_requested": dict(limit_requested_by_tf),
                    "rows_returned": dict(rows_returned_by_tf),
                    "rows_by_tf": dict(rows_returned_by_tf),
                }
                if rearm_fast_watch:
                    event_out["rearm_fast_watch"] = True
                if final_reason:
                    event_out["final_reason"] = final_reason
                if strategy_recheck_debug:
                    # Strategy-owned diagnostics (post-cleaning, repair fetch, etc.)
                    if "rows_used_after_clean" in strategy_recheck_debug:
                        event_out["rows_used_after_clean"] = strategy_recheck_debug.get("rows_used_after_clean")
                    if "used_fallback_fetch" in strategy_recheck_debug:
                        event_out["used_fallback_fetch"] = strategy_recheck_debug.get("used_fallback_fetch")
                    if "rows_by_tf" in strategy_recheck_debug and isinstance(strategy_recheck_debug.get("rows_by_tf"), dict):
                        event_out["rows_by_tf"] = dict(strategy_recheck_debug.get("rows_by_tf") or {})
                    if "sig_clean_rows" in strategy_recheck_debug:
                        event_out["sig_clean_rows"] = strategy_recheck_debug.get("sig_clean_rows")
                    if "vwap_clean_rows" in strategy_recheck_debug:
                        event_out["vwap_clean_rows"] = strategy_recheck_debug.get("vwap_clean_rows")
                    if "repair_reason" in strategy_recheck_debug:
                        event_out["repair_reason"] = strategy_recheck_debug.get("repair_reason")
                if outcome == "signal_emitted":
                    if emitted_signal_id:
                        event_out["signal_id"] = emitted_signal_id
                    else:
                        event_out["signal_id"] = None
                if outcome == "error":
                    event_out["error_code"] = error_code
                    event_out["error"] = error_detail
                logger.info(
                    "soft_deferral_recheck_outcome %s",
                    json.dumps(event_out, ensure_ascii=False, sort_keys=True),
                )

    def _get_default_symbols(self) -> List[str]:
        """
        Get default symbols with proper fallback logic.
        
        This is the single source of truth for symbol discovery when
        symbols are not explicitly provided via parameter.
        
        Fallback order:
        1. Config file: config['universe']['fixed_symbols']
        2. Environment variable: TRADING_SYMBOLS (comma-separated)
        3. Hardcoded defaults: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
        
        Returns:
            List of trading symbols
        """
        # Priority 1: Check config file
        config_symbols = self.config.get('universe', {}).get('fixed_symbols', [])
        if config_symbols and isinstance(config_symbols, list) and len(config_symbols) > 0:
            logger.info(f"[SYMBOL_DISCOVERY] Using {len(config_symbols)} symbols from config")
            return config_symbols
        
        # Priority 2: Check environment variable
        env_symbols = os.environ.get('TRADING_SYMBOLS', '').strip()
        if env_symbols:
            symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
            if symbols:
                logger.info(f"[SYMBOL_DISCOVERY] Using {len(symbols)} symbols from TRADING_SYMBOLS env var")
                return symbols
        
        # Priority 3: Use hardcoded defaults
        # Note: Using 3 major symbols as per issue requirements for cleaner default set.
        # Previous implementation had 8 symbols: BTC, ETH, SOL, BNB, ADA, DOT, LTC, AVAX
        # New default uses 3 major pairs (BTC, ETH, SOL) for sufficient fallback coverage
        # while keeping the default set minimal and maintainable.
        default_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        logger.info(f"[SYMBOL_DISCOVERY] Using {len(default_symbols)} hardcoded default symbols")
        return default_symbols
    
    async def is_data_layer_healthy(self) -> Dict[str, Any]:
        """
        Health check for data layer (Phase 1.5).
        
        Validates that the data layer is ready before ML initialization:
        - WebSocket connection is active
        - Subscriptions are successful  
        - At least one data packet has been received and processed
        
        Returns:
            Dict with 'healthy' bool and 'checks' dict with detailed status
        """
        logger.info("🏥 [HEALTH-CHECK] Performing data layer health check...")
        
        checks = {
            'websocket_connection': {'status': 'unknown', 'details': None},
            'subscriptions': {'status': 'unknown', 'details': None},
            'data_flow': {'status': 'unknown', 'details': None}
        }
        
        all_healthy = True
        
        # Check 1: WebSocket Connection
        if self.websocket_manager:
            try:
                # Check if any client is connected
                is_connected = self.websocket_manager.is_any_client_connected()
                if is_connected:
                    checks['websocket_connection'] = {
                        'status': 'healthy',
                        'details': 'At least one WebSocket client connected'
                    }
                    logger.info("   ✅ WebSocket connection: Active")
                else:
                    checks['websocket_connection'] = {
                        'status': 'unhealthy',
                        'details': 'No WebSocket clients connected'
                    }
                    all_healthy = False
                    logger.warning("   ⚠️ WebSocket connection: No active connections")
            except Exception as e:
                checks['websocket_connection'] = {
                    'status': 'error',
                    'details': str(e)
                }
                all_healthy = False
                logger.error(f"   ❌ WebSocket connection check failed: {e}")
        else:
            checks['websocket_connection'] = {
                'status': 'not_available',
                'details': 'WebSocket manager not initialized'
            }
            all_healthy = False
            logger.warning("   ⚠️ WebSocket manager not available")
        
        # Check 2: Subscriptions
        if self.websocket_manager:
            try:
                stream_count = self.websocket_manager.get_active_stream_count()
                if stream_count > 0:
                    checks['subscriptions'] = {
                        'status': 'healthy',
                        'details': f'{stream_count} active streams'
                    }
                    logger.info(f"   ✅ Subscriptions: {stream_count} active streams")
                else:
                    checks['subscriptions'] = {
                        'status': 'unhealthy',
                        'details': 'No active streams'
                    }
                    all_healthy = False
                    logger.warning("   ⚠️ Subscriptions: No active streams")
            except Exception as e:
                checks['subscriptions'] = {
                    'status': 'error',
                    'details': str(e)
                }
                logger.error(f"   ❌ Subscription check failed: {e}")
        else:
            checks['subscriptions'] = {
                'status': 'not_available',
                'details': 'WebSocket manager not initialized'
            }
            logger.warning("   ⚠️ Subscriptions check skipped (no WebSocket manager)")
        
        # Check 3: Data Flow
        if self.websocket_manager and hasattr(self.websocket_manager, 'collector'):
            try:
                # Check if collector has received any data for active symbols
                data_received = False
                sample_count = 0
                
                if self.active_symbols:
                    # Check first 3 symbols
                    for symbol in self.active_symbols[:3]:
                        # Try to get data for common timeframe
                        data = self.websocket_manager.get_latest_data(symbol, '1m')
                        if data and data.get('ohlcv'):
                            data_received = True
                            sample_count += 1
                
                if data_received:
                    checks['data_flow'] = {
                        'status': 'healthy',
                        'details': f'Data received for {sample_count} sample symbol(s)'
                    }
                    logger.info(f"   ✅ Data flow: Confirmed ({sample_count} symbols)")
                else:
                    checks['data_flow'] = {
                        'status': 'degraded',
                        'details': 'No data received yet (may need more time)'
                    }
                    # Don't fail - data might still be flowing in
                    logger.warning("   ⚠️ Data flow: No data received yet")
            except Exception as e:
                checks['data_flow'] = {
                    'status': 'error',
                    'details': str(e)
                }
                logger.error(f"   ❌ Data flow check failed: {e}")
        else:
            checks['data_flow'] = {
                'status': 'not_available',
                'details': 'WebSocket collector not available'
            }
            logger.warning("   ⚠️ Data flow check skipped (no collector)")
        
        # Summary
        if all_healthy:
            logger.info("🏥 [HEALTH-CHECK] ✅ Data layer is HEALTHY")
        else:
            logger.warning("🏥 [HEALTH-CHECK] ⚠️ Data layer has issues")
        
        return {
            'healthy': all_healthy,
            'checks': checks,
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def _initialize_ml_components(self, price_engine: Optional[Any] = None, regime_predictor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Initialize and connect ALL ML components with manifest-driven configuration.
        (YENİ YAPIYA UYGUN HALE GETİRİLDİ + Manifest entegrasyonu)
        """
        logger.info("🧠 [ML-INIT] Initializing ML system...")
        ml_components = []

        try:
            # Gerekli ML modüllerini import et
            from ml.feature_engineering import FeatureEngineeringPipeline
            from ml.price_predictor import AdvancedPricePredictionEngine
            from ml.regime_predictor import MLRegimePredictor
            from ml.reinforcement_learning import TradingRLAgent
            from ml.strategy_integration import MLStrategyIntegrationManager
            from ml.manifest_manager import ManifestManager
            logger.info("🧠 [ML-INIT] All ML modules imported successfully.")
        except ImportError as e:
            logger.error(f"🧠 [ML-INIT] Critical ML modules not found: {e}", exc_info=True)
            return {'success': False, 'reason': f'ML modules not available: {e}'}

        # Ana konfigürasyondan 'ml' bloğunu al
        ml_config = self.config.get('ml', {})
        
        # Initialize manifest manager first
        models_config = dict(self.config.get('models', {}))
        bundle_path = models_config.get('active_bundle', 'artifacts/legacy')
        fallback_bundle = models_config.get('fallback_bundle')

        manifest_path = Path(bundle_path) / "manifest.json"
        if not manifest_path.exists():
            if fallback_bundle:
                fallback_manifest_path = Path(fallback_bundle) / "manifest.json"
                if fallback_manifest_path.exists():
                    logger.warning(
                        "🧠 [ML-INIT] Active bundle manifest missing at %s; using fallback bundle %s",
                        manifest_path,
                        fallback_bundle,
                    )
                    bundle_path = fallback_bundle
                    models_config['active_bundle'] = bundle_path
                    manifest_path = fallback_manifest_path
                else:
                    logger.warning(
                        "🧠 [ML-INIT] Active bundle manifest missing at %s and fallback manifest not found at %s",
                        manifest_path,
                        fallback_manifest_path,
                    )
            else:
                logger.warning(
                    "🧠 [ML-INIT] Active bundle manifest missing at %s and no fallback bundle configured",
                    manifest_path,
                )

        # Persist any updates so downstream components receive the resolved path
        if models_config:
            self.config['models'] = models_config
        else:
            self.config['models'] = {'active_bundle': bundle_path}

        logger.info("🧠 [ML-INIT] Resolved model bundle path: %s", bundle_path)

        try:
            manifest_mgr = ManifestManager()
            manifest = manifest_mgr.load_manifest(bundle_path)

            logger.info(f"🧠 [ML-INIT] Using manifest version: {manifest.get('version')}")
            logger.info(f"🧠 [ML-INIT] Feature count: {manifest['feature_count']}")
            logger.info(f"🧠 [ML-INIT] Mode: {manifest.get('mode', 'unknown')}")

        except Exception as e:
            logger.warning(f"Failed to load manifest, using defaults: {e}")
            manifest = {'feature_count': 42, 'mode': 'legacy', 'version': '0.0-default'}

        # Pass manifest info to config for other components regardless of manifest status
        ml_config['active_bundle'] = bundle_path
        ml_config['manifest_version'] = manifest.get('version') if isinstance(manifest, dict) else None

        # 1. Özellik Mühendisliği Pijaması (Feature Engineering Pipeline)
        try:
            # Pass full config including models section for manifest access
            feature_config = ml_config.get('features', {})
            feature_config['models'] = self.config.get('models', {})
            feature_config['ml'] = ml_config  # For GEMMA enabled check
            
            self.feature_pipeline = FeatureEngineeringPipeline(config=feature_config)
            ml_components.append('feature_pipeline')
            logger.info("✅ Feature engineering pipeline initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FeatureEngineeringPipeline: {e}", exc_info=True)
            return {'success': False, 'reason': 'Failed to initialize FeatureEngineeringPipeline'}

        # 2. Fiyat Tahmin Motoru (Price Prediction Engine)
        # ✔️ DÜZELTME: Bu bileşene artık sadece 'price_prediction' alt bloğu verilir.
        price_pred_config = dict(ml_config.get('price_prediction', {}) or {})

        # Gemma adapter controls:
        # - Operational enable/disable: `ml.gemma.enabled` (single source of truth)
        # - Shadow mode precedence: `ml.price_prediction.shadow_mode` (explicit) > `ml.gemma.shadow_mode` (fallback)
        gemma_cfg = ml_config.get('gemma', {})
        if isinstance(gemma_cfg, dict):
            price_pred_config['gemma_enabled'] = bool(gemma_cfg.get('enabled', True))
            if 'shadow_mode' not in price_pred_config and 'shadow_mode' in gemma_cfg:
                price_pred_config['shadow_mode'] = bool(gemma_cfg.get('shadow_mode', False))
        if price_pred_config.get('enabled', True):
            try:
                self.price_engine = AdvancedPricePredictionEngine(
                    market_data_pipeline=self.market_data_pipeline,
                    feature_pipeline=self.feature_pipeline,
                    config=price_pred_config  # <-- DEĞİŞİKLİK BURADA
                )
                ml_components.append('price_engine')
                logger.info("✅ Price prediction engine initialized.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Price Prediction Engine: {e}", exc_info=True)
                self.price_engine = None
        else:
            self.price_engine = None
            logger.info("ℹ️ Price prediction is disabled in config.")

        # 3. Rejim Tahmincisi (Regime Predictor)
        # ✔️ DÜZELTME: Bu bileşene artık sadece 'regime_prediction' alt bloğu verilir + bundle path
        regime_pred_config = ml_config.get('regime_prediction', {})
        regime_pred_config['active_bundle'] = bundle_path  # Pass bundle path
        if regime_pred_config.get('enabled', True):
            try:
                self.regime_predictor = MLRegimePredictor(
                    feature_pipeline=self.feature_pipeline,
                    config=regime_pred_config
                )
                ml_components.append('regime_predictor')
                logger.info("✅ Regime predictor initialized.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Regime Predictor: {e}", exc_info=True)
                self.regime_predictor = None
        else:
            self.regime_predictor = None
            logger.info("ℹ️ Regime prediction is disabled in config.")

        # 4. Pekiştirmeli Öğrenme Ajanı (Reinforcement Learning Agent)
        # ✔️ DÜZELTME: Bu bileşene artık sadece 'reinforcement_learning' alt bloğu verilir + dynamic state_size
        rl_config = ml_config.get('reinforcement_learning', {})
        rl_config['active_bundle'] = bundle_path  # Pass bundle path
        legacy_rl_enabled = rl_config.get('enabled', True) and rl_config.get('legacy_dqn_enabled', False)
        self.rl_agent = None
        if legacy_rl_enabled:
            try:
                # Use manifest feature count for state size (manifest loaded above)
                state_size = manifest.get('rl_state_size', manifest.get('feature_count', 42))
                logger.info(f"🧠 [ML-INIT] Initializing RL agent with state_size={state_size}")
                
                self.rl_agent = TradingRLAgent(
                    state_size=state_size,
                    action_size=3,
                    config=rl_config
                )
                model_path = self.config.get('model_path', 'data/models')
                self.rl_agent.load_model(os.path.join(model_path, "rl_agent_final.pth"))
                if hasattr(self.rl_agent, 'set_inference_mode') and not rl_config.get('training_mode', False):
                    self.rl_agent.set_inference_mode(epsilon=rl_config.get('epsilon_inference', 0.0))
                    actual_training = getattr(self.rl_agent, 'training_mode', None)
                    actual_epsilon = getattr(self.rl_agent, 'epsilon', None)
                    logger.info("✅ RL Agent forced to inference mode at coordinator init")
                    logger.info(f"   - training_mode: {actual_training}")
                    logger.info(f"   - epsilon: {actual_epsilon}")
                    if actual_epsilon not in (0, 0.0):
                        raise ValueError(
                            f"RL inference mode enforcement failed: epsilon is {actual_epsilon}, expected 0.0"
                        )
                    if actual_training:
                        raise ValueError("RL inference mode enforcement failed: training_mode remained True")
                else:
                    logger.warning(
                        "⚠️ RL Agent lacks set_inference_mode support or training mode requested; ensure epsilon is safe."
                    )
                ml_components.append('rl_agent')
                logger.info("✅ Reinforcement learning agent initialized.")
                logger.info(f"   - RL Agent using hold_confidence_threshold: {self.rl_agent.hold_confidence_threshold}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Reinforcement Learning Agent: {e}", exc_info=True)
                self.rl_agent = None
        else:
            logger.info("ℹ️ Legacy DQN reinforcement learning is disabled.")

        # 5. PPO Adapter (optional soft RL guardrail)
        self.ppo_adapter = None
        if rl_config.get('ppo_enabled', False):
            if PPOTradingAdapter is None:
                logger.warning("⚠️ PPO adapter requested but module is unavailable (missing dependency).")
            else:
                try:
                    self.ppo_adapter = PPOTradingAdapter(
                        rl_config,
                        market_data_pipeline=self.market_data_pipeline,
                        feature_pipeline=self.feature_pipeline,
                    )
                    ml_components.append('ppo_adapter')
                    logger.info(f"✅ [PPO] Adapter initialized in ProductionCoordinator. Symbols: {rl_config.get('ppo_symbols')}")
                except Exception as e:
                    logger.error(f"❌ [PPO] Failed to initialize PPO adapter in ProductionCoordinator: {e}", exc_info=True)
                    self.ppo_adapter = None
        else:
            logger.info("ℹ️ [PPO] Adapter disabled in config (ProductionCoordinator).")

        # 6. ML Strateji Entegrasyon Yöneticisi
        try:
            # ✔️ DÜZELTME: Bu yöneticiye de tam 'ml_config' verilir.
            self.ml_integration = MLStrategyIntegrationManager(
                price_engine=self.price_engine,
                regime_predictor=self.regime_predictor,
                config=ml_config,
                market_data_pipeline=self.market_data_pipeline
            )
            ml_components.append('ml_integration')
            logger.info("✅ ML strategy integration manager initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Strategy Integration Manager: {e}", exc_info=True)
            self.ml_integration = None
        
        # 7. Bileşenleri sistemin diğer parçalarına bağla
        if self.strategy_coordinator:
            self.strategy_coordinator.ml_integration = self.ml_integration
            self.strategy_coordinator.feature_pipeline = self.feature_pipeline
            self.strategy_coordinator.rl_agent = self.rl_agent
            if hasattr(self.strategy_coordinator, 'ppo_adapter'):
                self.strategy_coordinator.ppo_adapter = self.ppo_adapter
            if hasattr(self.strategy_coordinator, 'on_ml_components_connected'):
                try:
                    self.strategy_coordinator.on_ml_components_connected()
                except Exception as exc:
                    logger.warning("⚠️ StrategyCoordinator ML hookup callback failed: %s", exc)
            logger.info("🔗 ML components connected to StrategyCoordinator.")
        if self.trading_engine:
            self.trading_engine.ml_integration = self.ml_integration
            self.trading_engine.rl_agent = self.rl_agent
            logger.info("🔗 ML components connected to LiveTradingEngine.")

        logger.info("🧠 [ML-INIT] ✅ ML SYSTEM INITIALIZATION COMPLETE")
        await self._ml_preflight_health_check()
        return {'success': True, 'components': ml_components}

    async def _ml_preflight_health_check(self) -> Dict[str, Any]:
        """
        Perform pre-flight health checks on ML system.
        
        Validates that ML models are loaded and can make basic predictions.
        This is the "ML model validation on startup" requirement.
        
        Returns:
            Health check results dictionary
        """
        logger.info("🧠 [ML-HEALTH-CHECK] Running pre-flight ML health checks...")
        
        results = {
            'overall_healthy': True,
            'checks': {},
            'warnings': []
        }
        
        # Check 1: Regime Predictor
        if hasattr(self, 'regime_predictor') and self.regime_predictor:
            try:
                # Create dummy data for health check
                dummy_data = pd.DataFrame({
                    'close': np.random.randn(100).cumsum() + 100,
                    'volume': np.random.rand(100) * 1000,
                    'high': np.random.randn(100).cumsum() + 102,
                    'low': np.random.randn(100).cumsum() + 98,
                    'open': np.random.randn(100).cumsum() + 100
                })
                
                # Try to make a prediction
                regime_result = await self.regime_predictor.predict_regime_transition(
                    ML_HEALTH_CHECK_SYMBOL, dummy_data
                )
                
                if regime_result and 'predicted_regime' in regime_result:
                    results['checks']['regime_predictor'] = {
                        'status': 'healthy',
                        'prediction': regime_result['predicted_regime'],
                        'confidence': regime_result.get('confidence', 0.0)
                    }
                    logger.info(f"   ✅ Regime Predictor: healthy (test prediction: {regime_result['predicted_regime']})")
                else:
                    results['checks']['regime_predictor'] = {'status': 'degraded', 'reason': 'No prediction returned'}
                    results['warnings'].append('Regime predictor returned no prediction')
                    logger.warning("   ⚠️ Regime Predictor: degraded - no prediction returned")
            except Exception as e:
                results['checks']['regime_predictor'] = {'status': 'unhealthy', 'error': str(e)}
                results['overall_healthy'] = False
                logger.error(f"   ❌ Regime Predictor: unhealthy - {e}")
        else:
            results['checks']['regime_predictor'] = {'status': 'not_available'}
            logger.info("   ℹ️ Regime Predictor: not available")
        
        # Check 2: ML Integration Manager
        if hasattr(self, 'ml_integration') and self.ml_integration:
            try:
                status = self.ml_integration.get_integration_status()
                results['checks']['ml_integration'] = {
                    'status': 'healthy',
                    'active': status.get('active', False)
                }
                logger.info(f"   ✅ ML Integration Manager: healthy")
            except Exception as e:
                results['checks']['ml_integration'] = {'status': 'unhealthy', 'error': str(e)}
                results['overall_healthy'] = False
                logger.error(f"   ❌ ML Integration Manager: unhealthy - {e}")
        else:
            results['checks']['ml_integration'] = {'status': 'not_available'}
            logger.info("   ℹ️ ML Integration Manager: not available")
        
        # Check 3: RL Agent
        if hasattr(self, 'rl_agent') and self.rl_agent:
            try:
                # Check if RL agent has memory set
                has_memory = self.rl_agent.memory is not None
                results['checks']['rl_agent'] = {
                    'status': 'healthy' if has_memory else 'degraded',
                    'has_memory': has_memory,
                    'epsilon': self.rl_agent.epsilon
                }
                if has_memory:
                    logger.info(f"   ✅ RL Agent: healthy (epsilon={self.rl_agent.epsilon:.4f})")
                else:
                    logger.warning("   ⚠️ RL Agent: degraded - no memory buffer")
                    results['warnings'].append('RL agent has no memory buffer')
            except Exception as e:
                results['checks']['rl_agent'] = {'status': 'unhealthy', 'error': str(e)}
                results['overall_healthy'] = False
                logger.error(f"   ❌ RL Agent: unhealthy - {e}")
        else:
            results['checks']['rl_agent'] = {'status': 'not_available'}
            logger.info("   ℹ️ RL Agent: not available")
        
        # Summary
        if results['overall_healthy']:
            logger.info("🧠 [ML-HEALTH-CHECK] ✅ All critical ML components are healthy")
        else:
            logger.error("🧠 [ML-HEALTH-CHECK] ❌ Some ML components failed health check")
        
        if results['warnings']:
            logger.warning(f"🧠 [ML-HEALTH-CHECK] ⚠️ Warnings: {', '.join(results['warnings'])}")
        
        return results
    
    async def initialize_core_systems(self,
                                      exchange_clients: Optional[Dict] = None,
                                      portfolio_config: Optional[Dict] = None,
                                      portfolio_value: Optional[float] = None,
                                      risk_config: Optional[RiskConfiguration] = None,
                                      mode: str = 'paper',
                                      trading_symbols: Optional[List[str]] = None,
                                      websocket_manager: Optional[Any] = None) -> Dict[str, Any]:
        """
        Initialize CORE systems only (Phase 1): Exchange, WebSocket, Data Pipeline, Risk, Portfolio.
        This method does NOT initialize ML components - that happens in initialize_ml_systems().
        
        Args:
            exchange_clients: Dictionary of exchange clients
            portfolio_value: Initial portfolio value in USD
            risk_config: RiskConfiguration object with all risk parameters
            mode: Trading mode ('paper' or 'live')
            trading_symbols: List of symbols to trade
            websocket_manager: Pre-initialized WebSocket manager
            
        Returns:
            Dict with 'success' and optional 'reason' keys
        """
        logger.info("="*70)
        logger.info("[PHASE 1] INITIALIZING CORE SYSTEMS")
        logger.info("="*70)
        
        try:
            # === STEP 1: ACCEPT EXTERNAL COMPONENTS ===
            if exchange_clients:
                self.exchange_clients = {k.lower(): v for k, v in exchange_clients.items()}
                logger.info(f"✓ Received {len(self.exchange_clients)} exchange client(s): {list(self.exchange_clients.keys())}")
            
            self.websocket_manager = websocket_manager

            # One-line runtime banner to prevent environment/operator mistakes.
            try:
                logger.warning(format_mode_banner(self.exchange_clients))
            except Exception as exc:
                logger.warning("[MODE-BANNER] failed to render: %s", exc)

            # Optional startup balance check (auth sanity): validates credentials + routing before trading.
            # Controlled by env var STARTUP_BALANCE_CHECK={off|auto|warn|required}. Default: auto (warn-only).
            async def _startup_balance_check() -> None:
                import os

                mode = (os.getenv("STARTUP_BALANCE_CHECK", "auto") or "auto").strip().lower()
                if mode in {"0", "false", "off", "no", "disabled"}:
                    return
                required = mode in {"1", "true", "on", "yes", "require", "required", "fail"}

                try:
                    from core.execution_env import get_bingx_env, is_real_execution_enabled

                    if not is_real_execution_enabled():
                        return
                    env_name = get_bingx_env()
                except Exception:
                    return

                bingx_client = (self.exchange_clients or {}).get("bingx")
                if not bingx_client or not hasattr(bingx_client, "get_bingx_balance"):
                    return

                try:
                    resp = await asyncio.to_thread(bingx_client.get_bingx_balance)
                    code = (resp or {}).get("code")
                    usdt_avail = None
                    if hasattr(bingx_client, "extract_bingx_usdt_available"):
                        usdt_avail = bingx_client.extract_bingx_usdt_available(resp)

                    if usdt_avail is None:
                        logger.info("[BALANCE-CHECK] exchange=bingx env=%s code=%s (USDT available not parsed)", env_name, code)
                    else:
                        logger.info("[BALANCE-CHECK] exchange=bingx env=%s code=%s usdt_available=%.4f", env_name, code, usdt_avail)
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                    if required:
                        logger.error("[BALANCE-CHECK] FAILED (required): %s", msg)
                        raise
                    logger.warning("[BALANCE-CHECK] failed (continuing): %s", msg)

            try:
                await _startup_balance_check()
            except Exception:
                return {"success": False, "reason": "startup_balance_check_failed"}
            if self.websocket_manager:
                logger.info("✓ WebSocket manager received from launcher (external).")
            else:
                logger.warning("⚠️ No WebSocket manager provided by launcher. Continuing without WebSocket (REST API fallback).")
            
            # === STEP 2: INITIALIZE MARKET DATA PIPELINE ===
            self.market_data_pipeline = MarketDataPipeline(
                exchanges=self.exchange_clients,
                config=self.config,
                websocket_manager=self.websocket_manager
            )
            logger.info("✓ Market data pipeline initialized")
 
            # === STEP 3: INITIALIZE PERFORMANCE MONITOR ===
            try:
                self.performance_monitor = PerformanceMonitor()
                logger.info("✓ Performance monitor initialized")
            except Exception as e:
                logger.warning(f"⚠️ PerformanceMonitor not available: {e}")
                logger.info("✓ Using fallback RealTimePerformanceMonitor")
                self.performance_monitor = RealTimePerformanceMonitor()
            
            # === STEP 4: PREPARE RISK MANAGER WITH STANDARDIZED CONFIG ===
            # Use provided risk_config or create default from config
            cfg_equity_override = None
            if isinstance(portfolio_config, dict) and portfolio_config.get('equity_usd') is not None:
                try:
                    cfg_equity_override = float(portfolio_config.get('equity_usd'))
                except (TypeError, ValueError):
                    cfg_equity_override = None

            if risk_config is None:
                config = self.config
                risk_params = dict(config.get('risk', {}) or {})
                if cfg_equity_override is not None:
                    risk_params['equity_usd'] = cfg_equity_override
                risk_config = RiskConfiguration(custom_limits=risk_params)
                logger.info("✓ Created RiskConfiguration from config file")
            else:
                logger.info("✓ Using provided RiskConfiguration")
            
            # Use provided portfolio_value or get from config
            if portfolio_value is None:
                config = self.config
                risk_params = dict(config.get('risk', {}) or {})
                if cfg_equity_override is not None:
                    risk_params['equity_usd'] = cfg_equity_override
                if risk_params.get('equity_usd') is None:
                    raise RuntimeError("Missing `risk.equity_usd` in configuration (or CAPITAL_USDT override).")
                portfolio_value = float(risk_params.get('equity_usd'))
                logger.info(f"✓ Portfolio value from config: ${portfolio_value:.2f}")
            else:
                logger.info(f"✓ Using provided portfolio value: ${portfolio_value:.2f}")
            
            # === STEP 5: INITIALIZE RISK MANAGER WITH STANDARDIZED CONFIG ===
            self.risk_manager = RiskManager(
                portfolio_value=portfolio_value,
                risk_config=risk_config,
                websocket_manager=self.websocket_manager,
                performance_monitor=self.performance_monitor
            )
            logger.info(f"✓ Risk manager initialized (portfolio value: ${self.risk_manager.portfolio_value:.2f})")

            # Emit one-time planner flag diagnostic for observability (safe for prod)
            env_planner = os.getenv("RISK_SIZE_PLANNER_ENABLED")
            config_planner = None
            try:
                cfg_dict = risk_config.to_dict() if hasattr(risk_config, "to_dict") else {}
                config_planner = (cfg_dict.get('risk') or {}).get('size_planner_enabled') if isinstance(cfg_dict, dict) else None
            except Exception:
                config_planner = None

            resolved_mode = 'active' if self.risk_manager._is_size_planner_enabled() else 'shadow'
            logger.info(
                "[RISK-PLANNER-FLAG] size_planner_flag_resolved",
                extra={
                    'env_value': env_planner,
                    'config_value': config_planner,
                    'resolved_mode': resolved_mode,
                },
            )

            # Startup health check for visibility (non-fatal in Sprint 1)
            try:
                health = self.risk_manager.run_health_check()
                logger.info("RiskManager startup health", extra={'health': health})
            except Exception as exc:
                logger.warning(f"⚠️ Failed to run RiskManager health check at startup: {exc}")

            # === STEP 6: INITIALIZE EXECUTION MANAGERS ===
            self.order_manager = SmartOrderManager(
                market_data_pipeline=self.market_data_pipeline,
                risk_manager=self.risk_manager,
                exchange_clients=self.exchange_clients
            )
            logger.info("✓ Order manager initialized with market_data_pipeline dependency")
            
            self.position_manager = AdvancedPositionManager(
                risk_manager=self.risk_manager,
                order_manager=self.order_manager,
                websocket_manager=self.websocket_manager
            )
            logger.info("✓ Position manager initialized and linked with OrderManager")

            # Link RiskManager to PositionManager for live PnL sourcing in scale-in gating
            try:
                self.risk_manager.set_pnl_provider(PositionManagerPnlProvider(self.position_manager))
                logger.info("✓ RiskManager PnL provider set from PositionManager")
            except Exception as exc:
                logger.warning(f"⚠️ Unable to set PnL provider on RiskManager: {exc}")
            
            # === STEP 7: INITIALIZE PORTFOLIO MANAGER ===
            self.portfolio_manager = PortfolioManager(
                risk_manager=self.risk_manager,
                performance_monitor=self.performance_monitor,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients
            )
            self.portfolio_manager.cfg = self.config
            logger.info("✓ Portfolio manager initialized")

            # === STEP 8: LINK MANAGERS ===
            self.portfolio_manager.set_execution_managers(self.order_manager, self.position_manager)
            # Note: set_dependencies is now optional as dependencies are injected in __init__
            # Kept for backward compatibility with other parts of the system
            logger.info("✓ All managers have been interlinked")

            # === STEP 9: VERIFY WEBSOCKET COLLECTOR READY ===
            if self.websocket_manager and hasattr(self.websocket_manager, 'is_collector_ready') and self.websocket_manager.is_collector_ready():
                logger.info("✓ WebSocket collector verified ready")
            else:
                logger.info("ℹ️ WebSocket manager/collector not ready - pipeline will use REST API only")
            
            # === STEP 10: INITIALIZE STRATEGY COORDINATOR ===
            self.strategy_coordinator = StrategyCoordinator(
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager,
                market_data_pipeline=self.market_data_pipeline,
                config=self.config,
                recheck_ready_callback=self.notify_recheck_ready,
            )
            logger.info("✓ Strategy coordinator initialized")

            if hasattr(self.strategy_coordinator, "start_fast_watcher"):
                try:
                    self.strategy_coordinator.start_fast_watcher()
                except Exception as exc:
                    logger.warning("[FAST-WATCH] Failed to start watcher: %s", exc)
            if hasattr(self.strategy_coordinator, "start_micro_gate_watcher"):
                try:
                    self.strategy_coordinator.start_micro_gate_watcher()
                except Exception as exc:
                    logger.warning("[MICRO-GATE] Failed to start watcher: %s", exc)
            
            # === STEP 11: INITIALIZE CIRCUIT BREAKER ===
            self.circuit_breaker = CircuitBreakerSystem(
                self.portfolio_manager,
                self.risk_manager
            )
            logger.info("✓ Circuit breaker system initialized")
            
            # === STEP 12: INITIALIZE LIVE TRADING ENGINE ===
            self.trading_engine = LiveTradingEngine(
                mode=mode,
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager,
                order_manager=self.order_manager,
                position_manager=self.position_manager,
                exchange_clients=self.exchange_clients,
                strategy_coordinator=self.strategy_coordinator,
                market_data_pipeline=self.market_data_pipeline
            )
            logger.info(f"✓ Live trading engine initialized (mode: {mode})")

            if (
                hasattr(self.position_manager, 'set_dispatch_notifier')
                and hasattr(self.trading_engine, 'trigger_coordinator_drain')
            ):
                self.position_manager.set_dispatch_notifier(
                    lambda: self.trading_engine.trigger_coordinator_drain(timeout=0.0)
                )
            
            # === STEP 13: SET ACTIVE SYMBOLS ===
            self.active_symbols = trading_symbols or self._get_default_symbols()
            logger.info(f"✓ Active symbols set: {len(self.active_symbols)} symbols")
            
            if self.trading_engine:
                self.trading_engine._cached_symbols = self.active_symbols
                logger.info(f"✓ Trading engine symbols cache set: {len(self.active_symbols)} symbols")
            
            if is_vst_fullbot_canary_enabled():
                try:
                    from strategies.vst_fullbot_canary import VstFullbotCanaryStrategy

                    canary_side = get_vst_fullbot_canary_side()
                    self.portfolio_manager.register_strategy(
                        "vst_fullbot_canary",
                        VstFullbotCanaryStrategy(side=canary_side),
                        initial_allocation=1.0,
                    )
                    logger.warning(
                        "[VST-FULLBOT-CANARY] Canary strategy registered | name=vst_fullbot_canary side=%s",
                        canary_side,
                    )
                except Exception as exc:
                    logger.error("[VST-FULLBOT-CANARY] Failed to register canary strategy: %s", exc, exc_info=True)
                    return {"success": False, "reason": f"canary_strategy_registration_failed: {exc}"}

            # === STEP 14: VALIDATE ACTIVE SYMBOLS ===
            if not self.active_symbols:
                logger.error("="*70)
                logger.error("❌ CRITICAL: NO ACTIVE SYMBOLS CONFIGURED!")
                logger.error("="*70)
                return {'success': False, 'reason': 'No active symbols configured'}
            else:
                logger.info("="*70)
                logger.info(f"✅ ACTIVE SYMBOLS CONFIGURED: {len(self.active_symbols)} symbols")
                logger.info("="*70)
                for idx, symbol in enumerate(self.active_symbols, 1):
                    logger.info(f"  {idx}. {symbol}")
                logger.info("="*70)
            
            # === STEP 15: PRIME DATA BUFFERS (CRITICAL FIX - Issue #259 followup) ===
            # This step fetches historical data ONCE at startup to warm up indicators
            # Previously this was done twice: once in trading engine start and once in preflight checks
            # Now it happens ONLY here in Phase 1, ensuring single data fetch
            logger.info("\n[STEP 15] Priming data buffers with historical data...")
            try:
                # Get timeframes from config
                timeframes_str = self.config.get('websocket', {}).get('stream_timeframes', '1m,5m,30m,1h,4h')
                if isinstance(timeframes_str, list):
                    timeframes = [item.strip() for sublist in (tf.split(',') for tf in timeframes_str) for item in sublist]
                else:
                    timeframes = [tf.strip() for tf in timeframes_str.split(',')]
                
                logger.info(f"[PRIME] Fetching historical data for {len(self.active_symbols)} symbols, {len(timeframes)} timeframes")
                await self.market_data_pipeline.prime_data_buffers_async(self.active_symbols, timeframes)
                logger.info("[PRIME] ✅ Historical data priming complete")
            except Exception as e:
                logger.error(f"[PRIME] ❌ Failed to prime data buffers: {e}", exc_info=True)
                logger.warning("[PRIME] Continuing without pre-primed data - may cause delays")
            
            # Mark core systems as initialized
            core_components = [
                'websocket_manager', 'performance_monitor', 'risk_manager',
                'portfolio_manager', 'strategy_coordinator', 'circuit_breaker', 'trading_engine'
            ]
            # Data priming completed successfully (tracked separately)
            data_priming_completed = True
            
            logger.info("="*70)
            logger.info("[PHASE 1] ✅ CORE SYSTEMS INITIALIZATION COMPLETE")
            logger.info("="*70)
            logger.info(f"Components initialized: {len(core_components)}")
            logger.info(f"Portfolio value: ${self.risk_manager.portfolio_value:.2f}")
            logger.info(f"Active symbols: {len(self.active_symbols)}")
            logger.info(f"Mode: {mode}")
            logger.info(f"Data priming: {'Completed' if data_priming_completed else 'Failed'}")
            logger.info("="*70)
            
            return {'success': True, 'components': core_components, 'active_symbols_count': len(self.active_symbols)}
            
        except Exception as e:
            logger.error("="*70)
            logger.error("[PHASE 1] ❌ CORE SYSTEMS INITIALIZATION FAILED")
            logger.error("="*70)
            logger.error(f"Error: {e}", exc_info=True)
            logger.error("="*70)
            return {'success': False, 'reason': str(e)}
    
    def _has_core_systems_initialized(self) -> bool:
        """
        Check if core systems are initialized.
        
        Used to determine if it's safe to mark the coordinator as fully initialized
        after ML initialization completes. Core systems (Phase 1) must be present
        before the coordinator can be considered operational.
        
        Returns:
            True if risk_manager (core system indicator) exists and is initialized
        """
        return hasattr(self, 'risk_manager') and self.risk_manager is not None
    
    def _mark_as_initialized_if_ready(self, status_message: str) -> None:
        """
        Mark coordinator as initialized if core systems are ready.
        
        This method encapsulates the pattern of checking if core systems are
        initialized and then setting the is_initialized flag with appropriate logging.
        
        Args:
            status_message: Status message to log when marking as initialized
        """
        if self._has_core_systems_initialized():
            self.is_initialized = True
            logger.info(f"✅ {status_message}")
    
    async def initialize_ml_systems(self, price_engine: Optional[Any] = None, regime_predictor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Initialize ML systems (Phase 2): ML prediction engines, RL agent, integrations.
        This method should ONLY be called after core systems are initialized and data layer is healthy.
        
        Args:
            price_engine: Pre-initialized price prediction engine
            regime_predictor: Pre-initialized regime predictor
            
        Returns:
            Dict with 'success' and optional 'reason' keys
        """
        logger.info("="*70)
        logger.info("[PHASE 2] INITIALIZING ML SYSTEMS")
        logger.info("="*70)
        
        ml_enabled = self.config.get('ml', {}).get('enabled', False)
        if not ml_enabled:
            logger.info("ℹ️ ML features disabled in config")
            return {'success': True, 'reason': 'ML disabled in config', 'components': []}
        
        if not self.active_symbols:
            logger.warning("⚠️ Cannot initialize ML without active symbols")
            return {'success': False, 'reason': 'No active symbols available'}
        
        try:
            ml_init_result = await self._initialize_ml_components(
                price_engine=price_engine,
                regime_predictor=regime_predictor
            )
            
            if ml_init_result.get('success'):
                logger.info("="*70)
                logger.info("[PHASE 2] ✅ ML SYSTEMS INITIALIZATION COMPLETE")
                logger.info("="*70)
                logger.info(f"Components: {', '.join(ml_init_result.get('components', []))}")
                logger.info("="*70)
                
                # Mark coordinator as fully initialized after successful ML init
                # (only if core systems were already initialized)
                self._mark_as_initialized_if_ready("Coordinator fully initialized (core + ML complete)")
                
                return ml_init_result
            else:
                logger.warning("="*70)
                logger.warning("[PHASE 2] ⚠️ ML INITIALIZATION PARTIAL")
                logger.warning("="*70)
                logger.warning(f"Reason: {ml_init_result.get('reason')}")
                logger.warning("Continuing with limited ML features")
                logger.warning("="*70)
                
                # Mark coordinator as initialized even with degraded ML
                # (only if core systems were already initialized)
                self._mark_as_initialized_if_ready("Coordinator initialized (core complete, ML degraded)")
                
                return ml_init_result
                
        except Exception as e:
            logger.error("="*70)
            logger.error("[PHASE 2] ❌ ML SYSTEMS INITIALIZATION FAILED")
            logger.error("="*70)
            logger.error(f"Error: {e}", exc_info=True)
            logger.error("Continuing without ML features")
            logger.error("="*70)
            
            # Mark coordinator as initialized even with ML failure
            # (only if core systems were already initialized)
            self._mark_as_initialized_if_ready("Coordinator initialized (core complete, ML failed)")
            
            return {'success': False, 'reason': str(e), 'components': []}
    
    async def _initialize_production_system(self) -> bool:
        """
        Legacy wrapper for backwards compatibility.
        
        This method wraps the public initialize_production_system() method
        and returns a boolean instead of a dict for legacy code compatibility.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        result = await self.initialize_production_system()
        return result.get('success', False)
    
    async def initialize_production_system(self, 
                                          exchange_clients: Optional[Dict] = None,
                                          portfolio_config: Optional[Dict] = None,
                                          mode: str = 'paper',
                                          trading_symbols: Optional[List[str]] = None,
                                          websocket_manager: Optional[Any] = None,
                                          # --- YENİ PARAMETRELER ---
                                          price_engine: Optional[Any] = None,
                                          regime_predictor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Orchestrates the phased initialization of the production system (DEPRECATED - use phased methods).
        
        This method now orchestrates the new phased initialization:
        - Phase 1: Core Systems (initialize_core_systems)
        - Phase 1.5: Data Layer Health Check (is_data_layer_healthy)
        - Phase 2: ML Systems (initialize_ml_systems)
        
        For new code, prefer calling the phase methods directly for better control.
        This method is kept for backward compatibility.
        
        Args:
            exchange_clients: Dictionary of exchange clients
            portfolio_config: Portfolio configuration
            mode: Trading mode ('paper' or 'live')
            trading_symbols: List of symbols to trade
            websocket_manager: Pre-initialized WebSocket manager
            price_engine: Pre-initialized price prediction engine
            regime_predictor: Pre-initialized regime predictor
            
        Returns:
            Dict with 'success', 'components', and optional 'reason' keys
        """
        logger.info("="*70)
        logger.info("INITIALIZING PRODUCTION SYSTEM (Phased Orchestration)")
        logger.info("="*70)
        
        try:
            # PHASE 1: Core Systems
            core_result = await self.initialize_core_systems(
                exchange_clients=exchange_clients,
                portfolio_config=portfolio_config,
                mode=mode,
                trading_symbols=trading_symbols,
                websocket_manager=websocket_manager
            )
            
            if not core_result.get('success'):
                logger.error(f"❌ Core systems initialization failed: {core_result.get('reason')}")
                return core_result
            
            # PHASE 1.5: Data Layer Health Check
            logger.info("\n" + "="*70)
            logger.info("[PHASE 1.5] DATA LAYER HEALTH CHECK")
            logger.info("="*70)
            
            health_result = await self.is_data_layer_healthy()
            
            if not health_result.get('healthy'):
                logger.warning("="*70)
                logger.warning("[PHASE 1.5] ⚠️ DATA LAYER NOT FULLY HEALTHY")
                logger.warning("="*70)
                logger.warning("ML initialization will proceed with degraded data layer")
                logger.warning("System may rely on REST API fallback")
                logger.warning("="*70)
                # Don't fail - continue with ML initialization
            else:
                logger.info("="*70)
                logger.info("[PHASE 1.5] ✅ DATA LAYER IS HEALTHY")
                logger.info("="*70)
            
            # PHASE 2: ML Systems (only if data layer is at least partially available)
            ml_result = await self.initialize_ml_systems(
                price_engine=price_engine,
                regime_predictor=regime_predictor
            )
            
            # Combine results
            all_components = core_result.get('components', []) + ml_result.get('components', [])
            
            # Mark as initialized
            self.is_initialized = True
            
            logger.info("="*70)
            logger.info("✅ PRODUCTION SYSTEM INITIALIZATION COMPLETE (All Phases)")
            logger.info("="*70)
            logger.info(f"Total components: {len(all_components)}")
            logger.info(f"Core: {len(core_result.get('components', []))}, ML: {len(ml_result.get('components', []))}")
            logger.info(f"Portfolio value: ${self.risk_manager.portfolio_value:.2f}")
            logger.info(f"Active symbols: {len(self.active_symbols)}")
            logger.info(f"Mode: {mode}")
            logger.info("="*70)
            
            return {
                'success': True,
                'components': all_components,
                'is_initialized': True,
                'active_symbols_count': len(self.active_symbols),
                'health_check': health_result
            }
            
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ PRODUCTION SYSTEM INITIALIZATION FAILED")
            logger.error("="*70)
            logger.error(f"Error: {e}", exc_info=True)
            logger.error("="*70)
            self.is_initialized = False
            return {'success': False, 'reason': str(e), 'is_initialized': False}
    
    @staticmethod
    def _vst_fullbot_canary_sanitize_order(order: Any) -> Dict[str, Any]:
        if not isinstance(order, dict):
            return {"type": str(type(order))}
        keep = (
            "id",
            "clientOrderId",
            "symbol",
            "type",
            "side",
            "status",
            "timestamp",
            "datetime",
            "price",
            "amount",
            "filled",
            "remaining",
            "reduceOnly",
            "positionSide",
        )
        return {k: order.get(k) for k in keep if k in order}

    async def _vst_fullbot_canary_fetch_exchange_state(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch exchange state for the canary symbol (open orders + positions).

        IMPORTANT: This is only intended to be called when BINGX_ENV=vst and real execution is enabled.
        """
        state: Dict[str, Any] = {"symbol": symbol, "open_orders": [], "positions": [], "errors": []}
        client = (self.exchange_clients or {}).get("bingx")
        if not client:
            state["errors"].append("missing_bingx_client")
            return state

        native_symbol = symbol
        to_native = getattr(client, "_get_bingx_native_symbol", None)
        if callable(to_native):
            try:
                native_symbol = str(to_native(symbol))
            except Exception:
                native_symbol = symbol

        try:
            fetch_open = getattr(getattr(client, "ex", None), "fetch_open_orders", None)
            if callable(fetch_open):
                open_orders = await asyncio.to_thread(fetch_open, native_symbol)
                state["open_orders"] = [self._vst_fullbot_canary_sanitize_order(o) for o in (open_orders or [])]
        except Exception as exc:
            state["errors"].append(f"fetch_open_orders_failed:{type(exc).__name__}:{str(exc)[:200]}")

        try:
            if hasattr(client, "get_bingx_positions"):
                resp = await asyncio.to_thread(client.get_bingx_positions, symbol)
                data = resp.get("data") if isinstance(resp, dict) else None
                if isinstance(data, list):
                    normalized: List[Dict[str, Any]] = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        normalized.append(
                            {
                                "symbol": item.get("symbol"),
                                "positionSide": item.get("positionSide"),
                                "positionAmt": item.get("positionAmt"),
                                "positionId": item.get("positionId"),
                            }
                        )
                    state["positions"] = normalized
        except Exception as exc:
            state["errors"].append(f"get_positions_failed:{type(exc).__name__}:{str(exc)[:200]}")

        return state

    async def _vst_fullbot_canary_preflight(self, symbol: str, *, allow_cleanup: bool) -> Dict[str, Any]:
        """
        Fail-fast preflight for a VST full-bot canary run.

        Default (safe) behavior: if any open orders/positions exist for symbol, return ok=False.
        If allow_cleanup=True, attempts to cancel open orders and close open positions (best-effort, symbol-scoped).
        """
        result: Dict[str, Any] = {
            "ok": False,
            "symbol": symbol,
            "allow_cleanup": bool(allow_cleanup),
            "errors": [],
            "before": {},
            "after": {},
            "cleanup": {"cancelled_orders": [], "closed_positions": []},
            "vst_balance": {},
        }

        if not is_real_execution_enabled():
            result["errors"].append("real_execution_not_enabled")
            return result
        if get_bingx_env() != "vst":
            result["errors"].append("BINGX_ENV_not_vst")
            return result

        client = (self.exchange_clients or {}).get("bingx")
        if not client:
            result["errors"].append("missing_bingx_client")
            return result

        try:
            await asyncio.to_thread(client.ensure_bingx_hedge_mode, symbol, True)
        except Exception as exc:
            result["errors"].append(f"hedge_mode_check_failed:{type(exc).__name__}:{str(exc)[:200]}")
            return result

        # Optional: Demo (VST) balance preflight + auto top-up.
        auto_topup_enabled = os.getenv("BINGX_VST_AUTO_TOPUP_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        if auto_topup_enabled:
            try:
                recv_window_ms = int(os.getenv("BINGX_VST_RECV_WINDOW_MS", "5000") or "5000")
            except (TypeError, ValueError):
                recv_window_ms = 5000
            try:
                threshold = float(os.getenv("BINGX_VST_TOPUP_THRESHOLD", "20000") or "20000")
            except (TypeError, ValueError):
                threshold = 20000.0
            try:
                topup_amount = float(os.getenv("BINGX_VST_TOPUP_AMOUNT", "100000") or "100000")
            except (TypeError, ValueError):
                topup_amount = 100000.0

            result["vst_balance"] = {
                "auto_topup_enabled": True,
                "threshold": threshold,
                "topup_amount": topup_amount,
                "recv_window_ms": recv_window_ms,
            }

            try:
                vst_balance_client = create_vst_balance_client_from_ccxt_bingx_client(
                    client,
                    recv_window_ms=recv_window_ms,
                    timeout_s=10.0,
                )
            except Exception as exc:
                result["errors"].append(f"vst_balance_client_init_failed:{type(exc).__name__}:{str(exc)[:200]}")
                return result

            try:
                balance_result = await asyncio.to_thread(vst_balance_client.get_vst_balance)
                balance = float(balance_result.balance)
                result["vst_balance"]["balance"] = balance
                result["vst_balance"]["balance_code"] = (balance_result.raw or {}).get("code")

                if balance < threshold:
                    logger.warning(
                        "[VST-TOPUP] balance below threshold: balance=%.4f threshold=%.4f; requesting topup=%.4f",
                        balance,
                        threshold,
                        topup_amount,
                    )
                    topup_resp = await asyncio.to_thread(vst_balance_client.apply_vst_topup, topup_amount)
                    result["vst_balance"]["topup_requested"] = True
                    result["vst_balance"]["topup_code"] = (topup_resp or {}).get("code")

                    # Re-check after top-up for operator clarity.
                    balance_after = await asyncio.to_thread(vst_balance_client.get_vst_balance)
                    result["vst_balance"]["balance_after"] = float(balance_after.balance)
            except BingxVstBalanceError as exc:
                result["errors"].append(f"vst_balance_failed:{str(exc)[:200]}")
                return result
            except Exception as exc:
                result["errors"].append(f"vst_balance_failed:{type(exc).__name__}:{str(exc)[:200]}")
                return result

        before = await self._vst_fullbot_canary_fetch_exchange_state(symbol)
        result["before"] = before

        open_orders = before.get("open_orders") if isinstance(before.get("open_orders"), list) else []
        open_positions = before.get("positions") if isinstance(before.get("positions"), list) else []
        has_open_orders = len(open_orders) > 0

        def _pos_amt(item: Dict[str, Any]) -> float:
            raw = item.get("positionAmt")
            try:
                return abs(float(raw))
            except (TypeError, ValueError):
                return 0.0

        open_pos_items = [p for p in open_positions if isinstance(p, dict) and _pos_amt(p) > 0]
        has_open_positions = len(open_pos_items) > 0

        if (has_open_orders or has_open_positions) and not allow_cleanup:
            result["errors"].append(
                f"dirty_state(open_orders={len(open_orders)}, open_positions={len(open_pos_items)})"
            )
            return result

        if allow_cleanup and has_open_orders:
            cancel_fn = getattr(client, "cancel_order", None)
            if not callable(cancel_fn):
                result["errors"].append("cancel_order_missing")
            else:
                for order in open_orders:
                    oid = (order or {}).get("id")
                    if not oid:
                        continue
                    try:
                        await asyncio.to_thread(cancel_fn, oid, symbol, {})
                        result["cleanup"]["cancelled_orders"].append({"order_id": oid, "ok": True})
                    except Exception as exc:
                        msg = str(exc).lower()
                        idempotent = any(
                            x in msg
                            for x in [
                                "already canceled",
                                "already cancelled",
                                "already closed",
                                "not found",
                                "does not exist",
                                "not exist",
                            ]
                        )
                        if idempotent:
                            result["cleanup"]["cancelled_orders"].append(
                                {"order_id": oid, "ok": True, "idempotent": True}
                            )
                        else:
                            result["cleanup"]["cancelled_orders"].append(
                                {"order_id": oid, "ok": False, "error": str(exc)[:200]}
                            )
                    await asyncio.sleep(0.1)

        if allow_cleanup and has_open_positions:
            create_fn = getattr(client, "create_order", None)
            if not callable(create_fn):
                result["errors"].append("create_order_missing")
            else:
                for pos in open_pos_items:
                    position_side = str(pos.get("positionSide") or "").upper().strip()
                    amt = _pos_amt(pos)
                    if amt <= 0:
                        continue
                    close_side = "sell" if position_side == "LONG" else "buy" if position_side == "SHORT" else None
                    if not close_side:
                        continue
                    params = {"reduceOnly": True, "positionSide": position_side}
                    try:
                        order = await asyncio.to_thread(
                            create_fn,
                            symbol=symbol,
                            side=close_side,
                            type_="market",
                            amount=amt,
                            price=None,
                            params=params,
                        )
                        oid = (order or {}).get("id") or (order or {}).get("orderId")
                        result["cleanup"]["closed_positions"].append(
                            {"positionSide": position_side, "qty": amt, "order_id": oid, "ok": True}
                        )
                    except Exception as exc:
                        result["cleanup"]["closed_positions"].append(
                            {"positionSide": position_side, "qty": amt, "ok": False, "error": str(exc)[:200]}
                        )
                    await asyncio.sleep(0.2)

        after = await self._vst_fullbot_canary_fetch_exchange_state(symbol)
        result["after"] = after

        after_open_orders = after.get("open_orders") if isinstance(after.get("open_orders"), list) else []
        after_positions = after.get("positions") if isinstance(after.get("positions"), list) else []
        after_open_pos_items = [p for p in after_positions if isinstance(p, dict) and _pos_amt(p) > 0]

        if after_open_orders or after_open_pos_items:
            result["errors"].append(
                f"dirty_state_after_cleanup(open_orders={len(after_open_orders)}, open_positions={len(after_open_pos_items)})"
            )
            return result

        result["ok"] = True
        return result

    async def _prod_canary_0_preflight(self, symbol: str, *, allow_cleanup: bool) -> Dict[str, Any]:
        """
        Fail-fast preflight for a production canary-0 run.

        Default (safe) behavior: if any open orders/positions exist for symbol, return ok=False.
        If allow_cleanup=True, attempts to cancel open orders and close open positions (best-effort, symbol-scoped).
        """
        result: Dict[str, Any] = {
            "ok": False,
            "symbol": symbol,
            "allow_cleanup": bool(allow_cleanup),
            "errors": [],
            "before": {},
            "after": {},
            "cleanup": {"cancelled_orders": [], "closed_positions": []},
        }

        if not is_real_execution_enabled():
            result["errors"].append("real_execution_not_enabled")
            return result
        if get_bingx_env() != "prod":
            result["errors"].append("BINGX_ENV_not_prod")
            return result

        client = (self.exchange_clients or {}).get("bingx")
        if not client:
            result["errors"].append("missing_bingx_client")
            return result

        try:
            await asyncio.to_thread(client.ensure_bingx_hedge_mode, symbol, True)
        except Exception as exc:
            result["errors"].append(f"hedge_mode_check_failed:{type(exc).__name__}:{str(exc)[:200]}")
            return result

        before = await self._vst_fullbot_canary_fetch_exchange_state(symbol)
        result["before"] = before

        open_orders = before.get("open_orders") if isinstance(before.get("open_orders"), list) else []
        open_positions = before.get("positions") if isinstance(before.get("positions"), list) else []
        has_open_orders = len(open_orders) > 0

        def _pos_amt(item: Dict[str, Any]) -> float:
            raw = item.get("positionAmt")
            try:
                return abs(float(raw))
            except (TypeError, ValueError):
                return 0.0

        open_pos_items = [p for p in open_positions if isinstance(p, dict) and _pos_amt(p) > 0]
        has_open_positions = len(open_pos_items) > 0

        if (has_open_orders or has_open_positions) and not allow_cleanup:
            result["errors"].append(
                f"dirty_state(open_orders={len(open_orders)}, open_positions={len(open_pos_items)})"
            )
            return result

        if allow_cleanup and has_open_orders:
            cancel_fn = getattr(client, "cancel_order", None)
            if not callable(cancel_fn):
                result["errors"].append("cancel_order_missing")
            else:
                for order in open_orders:
                    oid = (order or {}).get("id")
                    if not oid:
                        continue
                    try:
                        await asyncio.to_thread(cancel_fn, oid, symbol, {})
                        result["cleanup"]["cancelled_orders"].append({"order_id": oid, "ok": True})
                    except Exception as exc:
                        msg = str(exc).lower()
                        idempotent = any(
                            x in msg
                            for x in [
                                "already canceled",
                                "already cancelled",
                                "already closed",
                                "not found",
                                "does not exist",
                                "not exist",
                            ]
                        )
                        if idempotent:
                            result["cleanup"]["cancelled_orders"].append(
                                {"order_id": oid, "ok": True, "idempotent": True}
                            )
                        else:
                            result["cleanup"]["cancelled_orders"].append(
                                {"order_id": oid, "ok": False, "error": str(exc)[:200]}
                            )
                    await asyncio.sleep(0.1)

        if allow_cleanup and has_open_positions:
            create_fn = getattr(client, "create_order", None)
            if not callable(create_fn):
                result["errors"].append("create_order_missing")
            else:
                for pos in open_pos_items:
                    position_side = str(pos.get("positionSide") or "").upper().strip()
                    amt = _pos_amt(pos)
                    if amt <= 0:
                        continue
                    close_side = "sell" if position_side == "LONG" else "buy" if position_side == "SHORT" else None
                    if not close_side:
                        continue
                    params = {"reduceOnly": True, "positionSide": position_side}
                    try:
                        order = await asyncio.to_thread(
                            create_fn,
                            symbol=symbol,
                            side=close_side,
                            type_="market",
                            amount=amt,
                            price=None,
                            params=params,
                        )
                        oid = (order or {}).get("id") or (order or {}).get("orderId")
                        result["cleanup"]["closed_positions"].append(
                            {"positionSide": position_side, "qty": amt, "order_id": oid, "ok": True}
                        )
                    except Exception as exc:
                        result["cleanup"]["closed_positions"].append(
                            {"positionSide": position_side, "qty": amt, "ok": False, "error": str(exc)[:200]}
                        )
                    await asyncio.sleep(0.2)

        after = await self._vst_fullbot_canary_fetch_exchange_state(symbol)
        result["after"] = after

        after_open_orders = after.get("open_orders") if isinstance(after.get("open_orders"), list) else []
        after_positions = after.get("positions") if isinstance(after.get("positions"), list) else []
        after_open_pos_items = [p for p in after_positions if isinstance(p, dict) and _pos_amt(p) > 0]

        if after_open_orders or after_open_pos_items:
            result["errors"].append(
                f"dirty_state_after_cleanup(open_orders={len(after_open_orders)}, open_positions={len(after_open_pos_items)})"
            )
            return result

        result["ok"] = True
        return result

    def _write_vst_fullbot_canary_summary(self, summary: Dict[str, Any]) -> Optional[str]:
        try:
            out_dir = Path.cwd() / Path(get_vst_fullbot_canary_evidence_dir())
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"vst_fullbot_canary_summary_{ts}.json"
            latest_path = out_dir / "vst_fullbot_canary_summary_latest.json"

            payload = json.dumps(summary, indent=2, default=str, ensure_ascii=False)
            out_path.write_text(payload, encoding="utf-8")
            latest_path.write_text(payload, encoding="utf-8")
            logger.info("[VST-FULLBOT-CANARY] Wrote summary: %s", out_path)
            return str(out_path)
        except Exception as exc:
            logger.warning("[VST-FULLBOT-CANARY] Failed to write summary: %s", exc)
            return None

    def _write_prod_canary_0_summary(self, summary: Dict[str, Any]) -> Optional[str]:
        try:
            out_dir = Path.cwd() / Path(get_prod_canary_0_evidence_dir())
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"prod_canary_summary_{ts}.json"
            latest_path = out_dir / "prod_canary_summary_latest.json"

            payload = json.dumps(summary, indent=2, default=str, ensure_ascii=False)
            out_path.write_text(payload, encoding="utf-8")
            latest_path.write_text(payload, encoding="utf-8")
            logger.info("[PROD-CANARY-0] Wrote summary: %s", out_path)
            return str(out_path)
        except Exception as exc:
            logger.warning("[PROD-CANARY-0] Failed to write summary: %s", exc)
            return None

    async def run_production_loop(self, mode: str = 'paper', duration: Optional[float] = None,
                                  continuous: bool = False):
        # Emergency debug prints using ASCII to avoid encoding issues on Windows consoles.
        print(f"\n{'='*70}")
        print("EMERGENCY: run_production_loop() CALLED")
        print(f"   Time: {datetime.now(timezone.utc)}")
        print(f"   Mode: {mode}")
        print(f"   Duration: {duration}")
        print(f"   Continuous: {continuous}")
        print(f"   is_initialized: {self.is_initialized}")
        print(f"   is_running: {self.is_running}")
        print(f"   active_symbols: {self.active_symbols}")
        print(f"{'='*70}\n")
        
        import sys                              
        sys.stdout.flush()
        
        # Now try logger (keep ASCII for compatibility)
        logger.warning("[WARNING] run_production_loop() ENTERED")  # Use WARNING to ensure visibility
        logger.info("[INFO] run_production_loop() method ENTERED")
            
        try:
            logger.info("🔍 [DEBUG] Inside try block")
            
            if not self.is_initialized:
                logger.error("🔍 [DEBUG] NOT INITIALIZED - raising RuntimeError")
                raise RuntimeError("Production system not initialized. Call initialize_production_system() first.")
            
            logger.info("🔍 [DEBUG] Passed initialization check")
            logger.info("="*70)
            logger.info("STARTING PRODUCTION TRADING LOOP")
            logger.info("="*70)
            
            # Ensure the trading engine exists; it will be started after any VST canary preflight.
            logger.info("[DEBUG] Checking trading engine...")
            if not self.trading_engine:
                logger.error("[DEBUG] trading_engine is None!")
                raise RuntimeError("Trading engine not initialized!")

            logger.info(f"[DEBUG] trading_engine exists, state={self.trading_engine.state.value}")

            # Ensure is_running is True
            logger.info(f"[DEBUG] Current is_running = {self.is_running}")
            if not self.is_running:
                logger.warning("is_running was False, setting to True")
                self.is_running = True
            
            logger.info(f"[DEBUG] is_running now = {self.is_running}")

            # KONTROL: Price Prediction Engine varsa ve çalışmıyorsa, onu başlat.
            if hasattr(self, 'price_engine') and self.price_engine and not self.price_engine.is_running:
                logger.info("Activating Price Prediction Engine background loop...")
                # Zaman dilimlerini (timeframes) ML yapılandırmasından al
                ml_config = self.config.get('ml', {})
                prediction_timeframes = ml_config.get('prediction', {}).get('timeframes', ['5m', '15m', '1h'])
                
                # Arka plan görevini başlat
                asyncio.create_task(self.price_engine.start_prediction_loop(
                    symbols=self.active_symbols,
                    timeframes=prediction_timeframes
                ))
                logger.info("Price Prediction Engine loop started.")
            elif hasattr(self, 'price_engine') and self.price_engine and self.price_engine.is_running:
                logger.info("Price Prediction Engine is already running.")
            
            # Start queue monitoring task
            logger.info("[DEBUG] Creating queue monitoring task...")
            self._monitoring_task = asyncio.create_task(self._monitor_signal_queues())
            logger.info("[DEBUG] Queue monitoring task created")
            
            # Start watchdog task to detect if main loop stalls
            logger.info("[DEBUG] Creating watchdog task...")
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            logger.info("[DEBUG] Watchdog task created")

            # Start RL telemetry monitor if coordinator is ready
            if self.strategy_coordinator:
                logger.info("[DEBUG] Creating RL telemetry monitoring task...")
                self._rl_telemetry_task = asyncio.create_task(self._monitor_rl_telemetry())
                logger.info("[DEBUG] RL telemetry monitoring task created")
            else:
                logger.warning("[DEBUG] StrategyCoordinator missing; RL telemetry monitor not started.")
            
            logger.info("[DEBUG] About to print production loop info...")
            logger.info("\nProduction trading loop active")
            logger.info(f"   Mode: {mode}")
            logger.info(f"   Duration: {'Indefinite' if duration is None else f'{duration}s'}")
            logger.info(f"   Continuous Mode: {'ENABLED (Never stops, auto-recovers)' if continuous else 'DISABLED'}")
            logger.info(f"   Active Symbols: {len(self.active_symbols)}")

            # Active symbol health check
            logger.info(f"[DEBUG] Checking active_symbols: {self.active_symbols}")
            if not self.active_symbols:
                logger.error("No active symbols configured!")
                raise RuntimeError("active_symbols is empty! Cannot process any symbols.")

            logger.info("[DEBUG] active_symbols check passed")

            # -------------------------------------------------------------
            # Stage-3: Full-bot VST canary mode (feature-flagged, safe default OFF)
            # -------------------------------------------------------------
            vst_fullbot_canary_enabled = is_vst_fullbot_canary_enabled()
            vst_fullbot_canary_symbol: Optional[str] = None
            vst_fullbot_canary_started_at: Optional[datetime] = None
            vst_fullbot_canary_preflight: Optional[Dict[str, Any]] = None
            vst_fullbot_canary_max_closed_trades: Optional[int] = None
            vst_fullbot_canary_stop_reason: Optional[str] = None

            prod_canary_0_enabled = is_prod_canary_0_enabled()
            prod_canary_0_symbol: Optional[str] = None
            prod_canary_0_started_at: Optional[datetime] = None
            prod_canary_0_preflight: Optional[Dict[str, Any]] = None
            prod_canary_0_max_closed_trades: Optional[int] = None
            prod_canary_0_stop_reason: Optional[str] = None

            if vst_fullbot_canary_enabled:
                if prod_canary_0_enabled:
                    logger.error("[VST-FULLBOT-CANARY] Refusing to run: PROD_CANARY_0 is also enabled (choose one canary mode).")
                    self.is_running = False
                    return

                vst_fullbot_canary_started_at = datetime.now(timezone.utc)
                vst_fullbot_canary_max_closed_trades = get_vst_fullbot_canary_max_closed_trades()

                trading_mode = get_trading_mode()
                execution_backend = get_execution_backend()
                bingx_env = get_bingx_env()
                allow_cleanup = is_vst_fullbot_canary_cleanup_enabled()

                logger.warning(
                    "[VST-FULLBOT-CANARY] ENABLED | TRADING_MODE=%s EXECUTION_BACKEND=%s BINGX_ENV=%s allow_cleanup=%s max_closed_trades=%s",
                    trading_mode,
                    execution_backend,
                    bingx_env,
                    allow_cleanup,
                    vst_fullbot_canary_max_closed_trades,
                )

                if not is_real_execution_enabled():
                    logger.error(
                        "[VST-FULLBOT-CANARY] Refusing to run: real execution not enabled. "
                        "Set TRADING_MODE=live and EXECUTION_BACKEND=ccxt."
                    )
                    self.is_running = False
                    return

                if bingx_env != "vst":
                    logger.error("[VST-FULLBOT-CANARY] Refusing to run: BINGX_ENV must be vst.")
                    self.is_running = False
                    return

                if len(self.active_symbols) != 1:
                    logger.error(
                        "[VST-FULLBOT-CANARY] Refusing to run: must restrict to exactly 1 symbol (TRADING_SYMBOLS or config.universe.fixed_symbols). "
                        f"active_symbols={self.active_symbols}"
                    )
                    self.is_running = False
                    return

                vst_fullbot_canary_symbol = str(self.active_symbols[0])
                vst_fullbot_canary_preflight = await self._vst_fullbot_canary_preflight(
                    vst_fullbot_canary_symbol,
                    allow_cleanup=allow_cleanup,
                )

                if not vst_fullbot_canary_preflight.get("ok"):
                    vst_fullbot_canary_stop_reason = "preflight_failed"
                    logger.error(
                        "[VST-FULLBOT-CANARY] Preflight failed; aborting. errors=%s",
                        vst_fullbot_canary_preflight.get("errors"),
                    )

                    try:
                        if self.trading_engine:
                            await self.trading_engine.stop_live_trading()
                    except Exception:
                        pass

                    try:
                        final_state = await self._vst_fullbot_canary_fetch_exchange_state(vst_fullbot_canary_symbol)
                    except Exception:
                        final_state = {"errors": ["final_state_failed"]}

                    summary = {
                        "stage": "vst_fullbot_canary",
                        "ok": False,
                        "stop_reason": vst_fullbot_canary_stop_reason,
                        "started_at": (vst_fullbot_canary_started_at.isoformat() if vst_fullbot_canary_started_at else None),
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "symbol": vst_fullbot_canary_symbol,
                        "bingx": {
                            "ccxt_sandbox": bool(
                                getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "sandbox", False)
                                or getattr((self.exchange_clients or {}).get("bingx"), "bingx_env", None) == "vst"
                            ),
                            "ccxt_swap_url": str(
                                (
                                    (
                                        getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "urls", None)
                                        or {}
                                    )
                                    .get("api", {})
                                    .get("swap", "")
                                )
                            ),
                            "rest_base_url": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_rest_base_url", None),
                            "hedged": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_is_hedged", None),
                        },
                        "env": {
                            "TRADING_MODE": trading_mode,
                            "EXECUTION_BACKEND": execution_backend,
                            "BINGX_ENV": bingx_env,
                            "BINGX_NATIVE_HARD_STOP_ENABLED": os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", ""),
                            "BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED": os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", ""),
                        },
                        "preflight": vst_fullbot_canary_preflight,
                        "final_exchange_state": final_state,
                        "closed_positions": [],
                    }
                    self._write_vst_fullbot_canary_summary(summary)
                    self.is_running = False
                    return

            # -------------------------------------------------------------
            # Stage-4: Production Canary-0 (Hard stop only) - evidence mode
            # -------------------------------------------------------------
            if prod_canary_0_enabled:
                prod_canary_0_started_at = datetime.now(timezone.utc)
                prod_canary_0_max_closed_trades = get_prod_canary_0_max_closed_trades()

                trading_mode = get_trading_mode()
                execution_backend = get_execution_backend()
                bingx_env = get_bingx_env()
                allow_cleanup = is_prod_canary_0_cleanup_enabled()

                logger.warning(
                    "[PROD-CANARY-0] ENABLED | TRADING_MODE=%s EXECUTION_BACKEND=%s BINGX_ENV=%s allow_cleanup=%s max_closed_trades=%s",
                    trading_mode,
                    execution_backend,
                    bingx_env,
                    allow_cleanup,
                    prod_canary_0_max_closed_trades,
                )

                if not is_real_execution_enabled():
                    logger.error(
                        "[PROD-CANARY-0] Refusing to run: real execution not enabled. "
                        "Set TRADING_MODE=live and EXECUTION_BACKEND=ccxt."
                    )
                    self.is_running = False
                    return

                if bingx_env != "prod":
                    logger.error("[PROD-CANARY-0] Refusing to run: BINGX_ENV must be prod.")
                    self.is_running = False
                    return

                hard_stop_flag = os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
                trailing_flag = os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
                if not hard_stop_flag:
                    logger.error("[PROD-CANARY-0] Refusing to run: BINGX_NATIVE_HARD_STOP_ENABLED must be true.")
                    self.is_running = False
                    return
                if trailing_flag:
                    logger.error("[PROD-CANARY-0] Refusing to run: BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED must be false.")
                    self.is_running = False
                    return

                if len(self.active_symbols) != 1:
                    logger.error(
                        "[PROD-CANARY-0] Refusing to run: must restrict to exactly 1 symbol (TRADING_SYMBOLS or config.universe.fixed_symbols). "
                        f"active_symbols={self.active_symbols}"
                    )
                    self.is_running = False
                    return

                prod_canary_0_symbol = str(self.active_symbols[0])

                prod_canary_0_preflight = await self._prod_canary_0_preflight(
                    prod_canary_0_symbol,
                    allow_cleanup=allow_cleanup,
                )

                if not (prod_canary_0_preflight or {}).get("ok"):
                    prod_canary_0_stop_reason = "preflight_failed"
                    logger.error(
                        "[PROD-CANARY-0] Preflight failed; aborting. errors=%s",
                        (prod_canary_0_preflight or {}).get("errors"),
                    )

                    try:
                        if self.trading_engine:
                            await self.trading_engine.stop_live_trading()
                    except Exception:
                        pass

                    try:
                        final_state = await self._vst_fullbot_canary_fetch_exchange_state(prod_canary_0_symbol)
                    except Exception:
                        final_state = {"errors": ["final_state_failed"]}

                    summary = {
                        "stage": "prod_canary_0",
                        "ok": False,
                        "stop_reason": prod_canary_0_stop_reason,
                        "started_at": (prod_canary_0_started_at.isoformat() if prod_canary_0_started_at else None),
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "symbol": prod_canary_0_symbol,
                        "bingx": {
                            "ccxt_sandbox": bool(
                                getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "sandbox", False)
                                or getattr((self.exchange_clients or {}).get("bingx"), "bingx_env", None) == "vst"
                            ),
                            "ccxt_swap_url": str(
                                (
                                    (
                                        getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "urls", None)
                                        or {}
                                    )
                                    .get("api", {})
                                    .get("swap", "")
                                )
                            ),
                            "rest_base_url": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_rest_base_url", None),
                            "hedged": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_is_hedged", None),
                        },
                        "env": {
                            "TRADING_MODE": trading_mode,
                            "EXECUTION_BACKEND": execution_backend,
                            "BINGX_ENV": bingx_env,
                            "BINGX_NATIVE_HARD_STOP_ENABLED": os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", ""),
                            "BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED": os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", ""),
                            "PROD_CANARY_0_MAX_CLOSED_TRADES": prod_canary_0_max_closed_trades,
                            "PROD_CANARY_0_ALLOW_CLEANUP": os.getenv("PROD_CANARY_0_ALLOW_CLEANUP", ""),
                        },
                        "preflight": prod_canary_0_preflight,
                        "final_exchange_state": final_state,
                        "closed_positions": [],
                    }
                    self._write_prod_canary_0_summary(summary)
                    self.is_running = False
                    return

            # Start the trading engine (required for signal execution + position monitoring).
            if self.trading_engine and self.trading_engine.state.value != "running":
                logger.info("[DEBUG] Starting trading engine...")
                start_result = await self.trading_engine.start_live_trading(mode=mode)
                if not start_result.get("success"):
                    raise RuntimeError(f"Failed to start trading engine: {start_result.get('reason')}")
                logger.info("[DEBUG] Trading engine started (state=%s)", self.trading_engine.state.value)

            # Main loop setup
            logger.info("[DEBUG] Initializing loop variables...")
            start_time = vst_fullbot_canary_started_at or prod_canary_0_started_at or datetime.now(timezone.utc)
            last_recommendation_time = start_time
            recommendation_interval = 300  # Her 5 dakikada bir recommendations
            loop_iteration = 0

            logger.info(f"[DEBUG] Loop variables initialized: start_time={start_time}, loop_iteration={loop_iteration}")

            # Trading loop start log
            logger.info("")
            logger.info("="*70)
            logger.info("STARTING TRADING LOOP ITERATIONS")
            logger.info("="*70)
            logger.info(f"   Loop interval: {self.loop_interval}s")
            logger.info(f"   Symbols to process: {len(self.active_symbols)}")
            if duration:
                logger.info(f"   Will run for: {duration}s")
            else:
                logger.info(f"   Will run: Indefinitely")
            logger.info("="*70)
            logger.info("")
            
            # ✅ FORCE FLUSH before loop entry
            import sys
            sys.stdout.flush()
            print(f"[DEBUG] About to enter while loop, is_running={self.is_running}")
            sys.stderr.flush()
            
            logger.info(f"[DEBUG] About to enter while loop. is_running={self.is_running}")
            
            # ✅ CRITICAL CHECK: Verify is_running is True before loop
            if not self.is_running:
                logger.critical("[CRITICAL] is_running is FALSE before loop entry!")
                logger.critical("   This should never happen - is_running was just set to True earlier")
                raise RuntimeError("is_running unexpectedly False before loop entry")

            # =============================================================
            # ❌ KALDIRILDI: GEÇMİŞ VERİ ENJEKSİYONU (PRIMING)
            # Bu blok, LiveTradingEngine başlatılırken zaten yapıldığı için
            # gereksizdi ve sistem başlangıcını yavaşlatıyordu.
            # =============================================================
            
            # YENİ: Periyodik sağlık kontrolü için değişkenler
            last_health_check_time = time.monotonic()
            health_check_interval = 300 # 5 dakika

            while self.is_running:
                # ✅ ENHANCED: Always log loop entry at INFO level for visibility
                if loop_iteration == 0:
                    logger.info("[LOOP-START] Main trading loop entered successfully")

                print(f"[DEBUG] Loop iteration {loop_iteration + 1} processing symbols...")
                
                # Watchdog: Log heartbeat every 5 iterations
                if loop_iteration > 0 and loop_iteration % 5 == 0:
                    logger.info(f"[WATCHDOG] Loop heartbeat - {loop_iteration} iterations completed")
                
                try:
                    loop_iteration += 1
                    # ✅ DEĞİŞTİR: logger.debug → logger.info
                    logger.info(f"[ITERATION {loop_iteration}] Processing {len(self.active_symbols)} symbols...")
                    
                    # Check emergency conditions
                    if self.emergency_stop_triggered:
                        logger.critical("Emergency stop triggered - shutting down")
                        break
                    
                    # Check circuit breaker with timeout protection
                    try:
                        breaker_status = await asyncio.wait_for(
                            self.circuit_breaker.check_circuit_breaker(),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Circuit breaker check timeout - continuing")
                        breaker_status = {'tripped': False}
                    
                    if breaker_status.get('tripped'):
                        severity = breaker_status.get('severity', 'high')
                        
                        if continuous and severity != 'critical':
                            logger.warning(f"Circuit breaker tripped ({severity}): {breaker_status.get('message')}")
                            logger.warning("CONTINUOUS MODE: Bypassing non-critical breaker, continuing...")
                            await asyncio.sleep(10)
                            continue
                        else:
                            logger.critical(f"Circuit breaker tripped ({severity}): {breaker_status.get('message')}")
                            await self.handle_emergency_shutdown('circuit_breaker_tripped')
                            break
                    
                    # Check duration
                    if duration and not continuous:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        logger.info(f"[DEBUG] Duration check: elapsed={elapsed:.1f}s, duration={duration}s")
                        if elapsed >= duration:
                            logger.info(f"Duration {duration}s reached - stopping (elapsed: {elapsed:.1f}s)")
                            if vst_fullbot_canary_enabled and not vst_fullbot_canary_stop_reason:
                                vst_fullbot_canary_stop_reason = "duration_reached"
                            if prod_canary_0_enabled and not prod_canary_0_stop_reason:
                                prod_canary_0_stop_reason = "duration_reached"
                            break
                        else:
                            logger.info(f"[DEBUG] Duration check passed - continuing loop")
                    
                    # Process trading loop with WebSocket data
                    logger.info("[DEBUG] About to call _process_trading_loop()...")
                    await self._process_trading_loop()
                    logger.info("[DEBUG] _process_trading_loop() completed")

                    # Production Canary-0: if the engine requests abort (e.g., hard stop not placed), stop immediately.
                    if prod_canary_0_enabled:
                        abort_reason = getattr(self.trading_engine, "prod_canary_0_abort_reason", None) if self.trading_engine else None
                        if abort_reason and not prod_canary_0_stop_reason:
                            prod_canary_0_stop_reason = str(abort_reason)
                            logger.critical("[PROD-CANARY-0] Abort requested by engine: %s", abort_reason)
                            break

                    # VST full-bot canary: stop after N closed trades to keep the window minimal-risk.
                    if vst_fullbot_canary_enabled and vst_fullbot_canary_max_closed_trades:
                        try:
                            pm = getattr(self.trading_engine, "position_manager", None) if self.trading_engine else None
                            closed_positions = getattr(pm, "closed_positions", []) if pm else []
                            closed_count = len(closed_positions or [])
                        except Exception:
                            closed_count = 0

                        if closed_count >= int(vst_fullbot_canary_max_closed_trades):
                            vst_fullbot_canary_stop_reason = "max_closed_trades_reached"
                            logger.warning(
                                "[VST-FULLBOT-CANARY] Max closed trades reached (%s); stopping loop.",
                                vst_fullbot_canary_max_closed_trades,
                            )
                            break

                    # Production canary-0: stop after N closed trades to keep the window minimal-risk.
                    if prod_canary_0_enabled and prod_canary_0_max_closed_trades:
                        try:
                            pm = getattr(self.trading_engine, "position_manager", None) if self.trading_engine else None
                            closed_positions = getattr(pm, "closed_positions", []) if pm else []
                            closed_count = len(closed_positions or [])
                        except Exception:
                            closed_count = 0

                        if closed_count >= int(prod_canary_0_max_closed_trades):
                            prod_canary_0_stop_reason = "max_closed_trades_reached"
                            logger.warning(
                                "[PROD-CANARY-0] Max closed trades reached (%s); stopping loop.",
                                prod_canary_0_max_closed_trades,
                            )
                            break
                    
                    # YENİ: Periyodik indikatör sağlık kontrolü
                    current_time = time.monotonic()
                    if current_time - last_health_check_time >= health_check_interval:
                        await self.verify_indicator_health()
                        last_health_check_time = current_time

                    # Sleep between iterations, but check duration first
                    logger.debug(f"Trading loop iteration {loop_iteration} completed, sleeping {self.loop_interval}s")
                    
                    # If duration is set, calculate remaining time and don't sleep longer than needed
                    if duration and not continuous:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        remaining = duration - elapsed
                        if remaining <= 0:
                            logger.info(f"Duration {duration}s reached after processing - stopping")
                            if vst_fullbot_canary_enabled and not vst_fullbot_canary_stop_reason:
                                vst_fullbot_canary_stop_reason = "duration_reached"
                            if prod_canary_0_enabled and not prod_canary_0_stop_reason:
                                prod_canary_0_stop_reason = "duration_reached"
                            break
                        # Sleep for minimum of loop_interval or remaining time
                        sleep_time = min(self.loop_interval, remaining)
                        logger.debug(f"Sleeping for {sleep_time:.1f}s (remaining: {remaining:.1f}s)")
                        await self._sleep_with_recheck_wakeup(sleep_time)
                    else:
                        # No duration limit or continuous mode - use full loop_interval
                        await self._sleep_with_recheck_wakeup(self.loop_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received - stopping gracefully")
                    if vst_fullbot_canary_enabled and not vst_fullbot_canary_stop_reason:
                        vst_fullbot_canary_stop_reason = "keyboard_interrupt"
                    if prod_canary_0_enabled and not prod_canary_0_stop_reason:
                        prod_canary_0_stop_reason = "keyboard_interrupt"
                    break
                    
                except Exception as e:
                    logger.error(f"Error in production loop: {e}", exc_info=True)
                    
                    # Auto-recovery in continuous mode
                    if continuous:
                        logger.warning("CONTINUOUS MODE: Auto-recovering from error...")
                        
                        try:
                            if self.trading_engine and not self.trading_engine.is_running:
                                logger.info("Attempting to restart trading engine...")
                                restart_result = await self.trading_engine.start_live_trading(mode=mode)
                                if restart_result['success']:
                                    logger.info("Trading engine restarted successfully")
                                else:
                                    logger.error(f"Failed to restart trading engine: {restart_result.get('reason')}")
                        except Exception as restart_error:
                            logger.error(f"Error during auto-recovery: {restart_error}")
                        
                        await asyncio.sleep(5)
                        continue
                    else:
                        # Original behavior for non-continuous mode
                        if self.config.get('emergency', {}).get('enable_circuit_breaker', True):
                            await self.handle_emergency_shutdown('system_error')
                            break
                        else:
                            await asyncio.sleep(5)
                            continue
            
            # Shutdown
            logger.info("\nShutting down production trading loop...")
            await self.trading_engine.stop_live_trading()

            if vst_fullbot_canary_enabled and vst_fullbot_canary_symbol:
                ended_at = datetime.now(timezone.utc)
                try:
                    final_state = await self._vst_fullbot_canary_fetch_exchange_state(vst_fullbot_canary_symbol)
                except Exception:
                    final_state = {"errors": ["final_exchange_state_failed"]}

                pm = getattr(self.trading_engine, "position_manager", None) if self.trading_engine else None
                closed_positions = getattr(pm, "closed_positions", []) if pm else []

                trades: List[Dict[str, Any]] = []
                for pos in closed_positions or []:
                    if not isinstance(pos, dict):
                        continue
                    trades.append(
                        {
                            "position_id": pos.get("position_id"),
                            "trade_id": pos.get("trade_id"),
                            "symbol": pos.get("symbol"),
                            "side": pos.get("side"),
                            "amount": pos.get("amount"),
                            "entry_price": pos.get("entry_price"),
                            "exit_price": pos.get("exit_price"),
                            "realized_pnl": pos.get("realized_pnl"),
                            "return_pct": pos.get("return_pct"),
                            "exit_reason": pos.get("exit_reason"),
                            "opened_at": str(pos.get("opened_at") or ""),
                            "closed_at": str(pos.get("closed_at") or ""),
                            "entry_order_id": pos.get("entry_order_id"),
                            "exit_order_id": pos.get("exit_order_id"),
                            "native_hard_stop": {
                                "order_id": pos.get("native_hard_stop_order_id"),
                                "stop_price": pos.get("native_hard_stop_stop_price"),
                                "working_type": pos.get("native_hard_stop_working_type"),
                                "position_side": pos.get("native_hard_stop_position_side"),
                                "qty": pos.get("native_hard_stop_qty"),
                            },
                            "native_trailing_stop": {
                                "order_id": pos.get("native_trailing_stop_order_id"),
                                "trailing_percent": pos.get("native_trailing_stop_trailing_percent"),
                                "working_type": pos.get("native_trailing_stop_working_type"),
                                "position_side": pos.get("native_trailing_stop_position_side"),
                                "qty": pos.get("native_trailing_stop_qty"),
                            },
                        }
                    )

                summary = {
                    "stage": "vst_fullbot_canary",
                    "ok": bool(trades),
                    "stop_reason": vst_fullbot_canary_stop_reason or "loop_ended",
                    "started_at": (vst_fullbot_canary_started_at.isoformat() if vst_fullbot_canary_started_at else None),
                    "ended_at": ended_at.isoformat(),
                    "symbol": vst_fullbot_canary_symbol,
                    "closed_trades_count": len(trades),
                    "bingx": {
                        "ccxt_sandbox": bool(
                            getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "sandbox", False)
                            or getattr((self.exchange_clients or {}).get("bingx"), "bingx_env", None) == "vst"
                        ),
                        "ccxt_swap_url": str(
                            (
                                (
                                    getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "urls", None)
                                    or {}
                                )
                                .get("api", {})
                                .get("swap", "")
                            )
                        ),
                        "rest_base_url": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_rest_base_url", None),
                        "hedged": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_is_hedged", None),
                    },
                    "env": {
                        "TRADING_MODE": get_trading_mode(),
                        "EXECUTION_BACKEND": get_execution_backend(),
                        "BINGX_ENV": get_bingx_env(),
                        "BINGX_NATIVE_HARD_STOP_ENABLED": os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", ""),
                        "BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED": os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", ""),
                        "VST_FULLBOT_CANARY_MAX_CLOSED_TRADES": vst_fullbot_canary_max_closed_trades,
                        "VST_FULLBOT_CANARY_SIDE": os.getenv("VST_FULLBOT_CANARY_SIDE", ""),
                        "VST_FULLBOT_CANARY_ALLOW_CLEANUP": os.getenv("VST_FULLBOT_CANARY_ALLOW_CLEANUP", ""),
                    },
                    "preflight": vst_fullbot_canary_preflight,
                    "trades": trades,
                    "final_exchange_state": final_state,
                }
                self._write_vst_fullbot_canary_summary(summary)

            if prod_canary_0_enabled and prod_canary_0_symbol:
                ended_at = datetime.now(timezone.utc)
                try:
                    final_state = await self._vst_fullbot_canary_fetch_exchange_state(prod_canary_0_symbol)
                except Exception:
                    final_state = {"errors": ["final_exchange_state_failed"]}

                pm = getattr(self.trading_engine, "position_manager", None) if self.trading_engine else None
                closed_positions = getattr(pm, "closed_positions", []) if pm else []

                trades: List[Dict[str, Any]] = []
                for pos in closed_positions or []:
                    if not isinstance(pos, dict):
                        continue
                    trades.append(
                        {
                            "position_id": pos.get("position_id"),
                            "trade_id": pos.get("trade_id"),
                            "symbol": pos.get("symbol"),
                            "side": pos.get("side"),
                            "amount": pos.get("amount"),
                            "entry_price": pos.get("entry_price"),
                            "exit_price": pos.get("exit_price"),
                            "realized_pnl": pos.get("realized_pnl"),
                            "return_pct": pos.get("return_pct"),
                            "exit_reason": pos.get("exit_reason"),
                            "opened_at": str(pos.get("opened_at") or ""),
                            "closed_at": str(pos.get("closed_at") or ""),
                            "entry_order_id": pos.get("entry_order_id"),
                            "exit_order_id": pos.get("exit_order_id"),
                            "native_hard_stop": {
                                "order_id": pos.get("native_hard_stop_order_id"),
                                "stop_price": pos.get("native_hard_stop_stop_price"),
                                "working_type": pos.get("native_hard_stop_working_type"),
                                "position_side": pos.get("native_hard_stop_position_side"),
                                "qty": pos.get("native_hard_stop_qty"),
                            },
                            "native_hard_stop_place_failed": pos.get("native_hard_stop_place_failed"),
                            "native_hard_stop_place_failed_reason": pos.get("native_hard_stop_place_failed_reason"),
                            "native_hard_stop_place_failed_error_code": pos.get("native_hard_stop_place_failed_error_code"),
                            "native_trailing_stop": {
                                "order_id": pos.get("native_trailing_stop_order_id"),
                                "trailing_percent": pos.get("native_trailing_stop_trailing_percent"),
                                "working_type": pos.get("native_trailing_stop_working_type"),
                                "position_side": pos.get("native_trailing_stop_position_side"),
                                "qty": pos.get("native_trailing_stop_qty"),
                            },
                            "prod_canary_0_abort_reason": pos.get("prod_canary_0_abort_reason"),
                        }
                    )

                abort_reason = getattr(self.trading_engine, "prod_canary_0_abort_reason", None) if self.trading_engine else None
                hard_stop_missing = any(not (t.get("native_hard_stop") or {}).get("order_id") for t in trades)
                hard_stop_failed = any((t.get("native_hard_stop_place_failed") is True) for t in trades)
                ok = bool(trades) and not abort_reason and not hard_stop_missing and not hard_stop_failed

                summary = {
                    "stage": "prod_canary_0",
                    "ok": ok,
                    "stop_reason": prod_canary_0_stop_reason or (str(abort_reason) if abort_reason else "loop_ended"),
                    "started_at": (prod_canary_0_started_at.isoformat() if prod_canary_0_started_at else None),
                    "ended_at": ended_at.isoformat(),
                    "symbol": prod_canary_0_symbol,
                    "closed_trades_count": len(trades),
                    "abort_reason": abort_reason,
                    "invariants": {
                        "native_hard_stop_required": True,
                        "native_hard_stop_present_for_all_trades": (not hard_stop_missing),
                        "native_hard_stop_place_failed": hard_stop_failed,
                    },
                    "bingx": {
                        "ccxt_sandbox": bool(
                            getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "sandbox", False)
                            or getattr((self.exchange_clients or {}).get("bingx"), "bingx_env", None) == "vst"
                        ),
                        "ccxt_swap_url": str(
                            (
                                (
                                    getattr(getattr((self.exchange_clients or {}).get("bingx"), "ex", None), "urls", None)
                                    or {}
                                )
                                .get("api", {})
                                .get("swap", "")
                            )
                        ),
                        "rest_base_url": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_rest_base_url", None),
                        "hedged": getattr((self.exchange_clients or {}).get("bingx"), "_bingx_is_hedged", None),
                    },
                    "env": {
                        "TRADING_MODE": get_trading_mode(),
                        "EXECUTION_BACKEND": get_execution_backend(),
                        "BINGX_ENV": get_bingx_env(),
                        "BINGX_NATIVE_HARD_STOP_ENABLED": os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", ""),
                        "BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED": os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", ""),
                        "PROD_CANARY_0_MAX_CLOSED_TRADES": prod_canary_0_max_closed_trades,
                        "PROD_CANARY_0_ALLOW_CLEANUP": os.getenv("PROD_CANARY_0_ALLOW_CLEANUP", ""),
                    },
                    "preflight": prod_canary_0_preflight,
                    "trades": trades,
                    "final_exchange_state": final_state,
                }
                self._write_prod_canary_0_summary(summary)

            self.is_running = False
             
            logger.info("Production trading loop stopped")
            
        except Exception as e:
            logger.error(f"Critical error in production loop: {e}", exc_info=True)
            self.is_running = False
            await self.handle_emergency_shutdown('critical_error')
    
    async def handle_emergency_shutdown(self, reason: str):
        """
        Emergency shutdown protocol.
        
        Args:
            reason: Reason for emergency shutdown
        """
        try:
            logger.critical("="*70)
            logger.critical("EMERGENCY SHUTDOWN INITIATED")
            logger.critical(f"Reason: {reason}")
            logger.critical("="*70)
            
            self.emergency_stop_triggered = True
            
            # Step 1: Stop new signals
            logger.critical("Step 1: Stopping new signal processing...")
            self.is_running = False
            
            # Step 2: Cancel pending orders
            logger.critical("Step 2: Cancelling pending orders...")
            # Would iterate through active orders and cancel them
            
            # Step 3: Close positions (if configured)
            close_method = self.config.get('emergency', {}).get('emergency_close_method', 'market')
            logger.critical(f"Step 3: Closing positions using {close_method} method...")
            
            # Get active positions
            if self.trading_engine:
                active_positions = list(self.trading_engine.active_positions.keys())
                
                if active_positions:
                    logger.critical(f"Closing {len(active_positions)} active positions...")
                    
                    # Execute emergency protocol via circuit breaker
                    if self.circuit_breaker:
                        await self.circuit_breaker.execute_emergency_protocol('close_all', active_positions)
                else:
                    logger.critical("No active positions to close")
            
            # Step 4: Stop trading engine
            if self.trading_engine:
                logger.critical("Step 4: Stopping trading engine...")
                await self.trading_engine.stop_live_trading()
            
            # Step 5: Close WebSocket connections
            if self.websocket_manager:
                logger.critical("Step 5: Closing WebSocket connections...")
                await self.websocket_manager.close()
            
            # Step 6: Save state
            logger.critical("Step 6: Saving system state...")
            state = self.get_system_state()
            # Could save state to file here
            
            # Step 7: Generate emergency report
            logger.critical("Step 7: Generating emergency report...")
            report = self._generate_emergency_report(reason)
            
            logger.critical("\n" + "="*70)
            logger.critical("EMERGENCY SHUTDOWN COMPLETE")
            logger.critical("="*70)
            
            # Could send alert notification here
            
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}", exc_info=True)
    
    def _generate_emergency_report(self, reason: str) -> Dict[str, Any]:
        """Generate emergency shutdown report."""
        report = {
            'timestamp': datetime.now(timezone.utc),
            'reason': reason,
            'system_state': self.get_system_state()
        }
        
        if self.trading_engine:
            report['engine_status'] = self.trading_engine.get_engine_status()
        
        if self.portfolio_manager:
            report['portfolio_state'] = self.portfolio_manager.portfolio_state
        
        return report
    
    async def submit_signal(self, signal: Dict) -> Dict[str, Any]:
        """
        Submit trading signal to the system.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Submission result
        """
        try:
            if not self.is_running:
                return {
                    'success': False,
                    'reason': 'Trading system not running'
                }
            
            # Process through strategy coordinator
            result = await self.strategy_coordinator.process_strategy_signal(
                strategy_name=signal.get('strategy', 'unknown'),
                signal=signal
            )
            
            if result['status'] == 'accepted':
                # Get signal_id from coordinator result
                signal_id = result.get('signal_id')
                
                # [STAGE 1: GENERATED] Mark signal as generated
                self._track_signal_lifecycle(signal_id, 'generated', {'symbol': signal.get('symbol'), 'strategy': signal.get('strategy')})
                logger.info(f"[STAGE:GENERATED] Signal {signal_id} for {signal.get('symbol')}")
                
                # [STAGE 2: VALIDATED] Signal passed validation
                enriched_signal = result['enriched_signal']
                self._track_signal_lifecycle(signal_id, 'validated', {'reason': 'passed_all_checks'})
                logger.info(f"[STAGE:VALIDATED] Signal {signal_id} validated")
                
                # [STAGE 3: QUEUED] Signal added to StrategyCoordinator queue (already done in coordinator)
                self._track_signal_lifecycle(signal_id, 'queued', {'queue': 'strategy_coordinator'})
                logger.info(f"[STAGE:QUEUED] Signal {signal_id} in StrategyCoordinator queue")
                
                # Signal is now queued in StrategyCoordinator, will be bridged to LiveTradingEngine
                logger.info(f"✅ [SIGNAL-ACCEPTED] Signal {signal_id} accepted by StrategyCoordinator")
                logger.info(f"💡 [SIGNAL-QUEUED] {signal.get('strategy', 'unknown').upper()} signal for {signal.get('symbol')} queued in StrategyCoordinator")
                
                # Log queue state for monitoring
                coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
                logger.info(f"📊 [QUEUE-STATE] StrategyCoordinator queue size: {coordinator_queue_size}")
                
                return {'success': True, 'signal_id': signal_id}
            else:
                logger.warning(f"Signal rejected: {result.get('reason')}")
                # For rejected signals, generate a temporary ID for tracking
                signal_id = f"{signal.get('strategy', 'unknown')}_{signal.get('symbol', 'UNKNOWN')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
                # [STAGE 1: GENERATED] Mark signal as generated for rejected signals
                self._track_signal_lifecycle(signal_id, 'generated', {'symbol': signal.get('symbol'), 'strategy': signal.get('strategy')})
                # [STAGE 2: REJECTED] Mark signal as rejected
                self._track_signal_lifecycle(signal_id, 'rejected', {'reason': result.get('reason')})
                return {'success': False, 'reason': result.get('reason')}
                
        except Exception as e:
            logger.error(f"Error submitting signal: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _track_signal_lifecycle(self, signal_id: str, stage: str, details: Dict = None):
        """
        Track signal through its lifecycle stages.
        
        Stages: generated -> validated -> queued -> forwarded -> executed
        
        Args:
            signal_id: Unique signal identifier
            stage: Current stage
            details: Additional details about this stage
        """
        if signal_id not in self.signal_lifecycle:
            self.signal_lifecycle[signal_id] = {
                'stages': [],
                'created_at': datetime.now(timezone.utc)
            }
        
        self.signal_lifecycle[signal_id]['stages'].append({
            'stage': stage,
            'timestamp': datetime.now(timezone.utc),
            'details': details or {}
        })
        
        # Keep only last 100 signals to avoid memory issues
        if len(self.signal_lifecycle) > 100:
            oldest_key = min(self.signal_lifecycle.keys(), 
                           key=lambda k: self.signal_lifecycle[k]['created_at'])
            del self.signal_lifecycle[oldest_key]
    
    def register_strategy(self, strategy_name: str, strategy_instance: Any, 
                         initial_allocation: float = 0.25) -> Dict[str, Any]:
        """
        Register a trading strategy with the system.
        
        Args:
            strategy_name: Unique strategy identifier
            strategy_instance: Strategy instance
            initial_allocation: Initial capital allocation
            
        Returns:
            Registration result
        """
        try:
            if not self.is_initialized:
                return {
                    'success': False,
                    'reason': 'System not initialized'
                }
            
            # Store strategy reference in coordinator
            self.strategies[strategy_name] = strategy_instance

            # Auto-wire market_data_pipeline if missing and supported
            if hasattr(strategy_instance, 'market_data_pipeline'):
                if getattr(strategy_instance, 'market_data_pipeline', None) is None:
                    strategy_instance.market_data_pipeline = self.market_data_pipeline
                    logger.info(f"[STRATEGY-WIRE] Injected market_data_pipeline into {strategy_name}")
                else:
                    logger.debug(f"[STRATEGY-WIRE] Skipped injection for {strategy_name}, pipeline already set")
            
            # Cache strategy capabilities to avoid repeated inspection
            capabilities = {
                'supports_regime_data': False,
                'is_async': False
            }
            
            # Check if strategy has signal method and supports regime_data
            if hasattr(strategy_instance, 'signal'):
                sig = inspect.signature(strategy_instance.signal)
                capabilities['supports_regime_data'] = 'regime_data' in sig.parameters
            
            # Check if strategy has async generate_signal method
            if hasattr(strategy_instance, 'generate_signal'):
                capabilities['is_async'] = inspect.iscoroutinefunction(strategy_instance.generate_signal)
            
            self.strategy_capabilities[strategy_name] = capabilities
            
            result = self.portfolio_manager.register_strategy(
                strategy_name=strategy_name,
                strategy_instance=strategy_instance,
                initial_allocation=initial_allocation
            )
            
            logger.info(f"Strategy registered: {strategy_name} with {initial_allocation*100}% allocation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return {'success': False, 'reason': str(e)}
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        state = {
            'timestamp': datetime.now(timezone.utc),
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'emergency_stop': self.emergency_stop_triggered,
            'active_symbols': self.active_symbols,
            'processed_symbols': self.processed_symbols_count
        }
        
        if self.trading_engine:
            state['trading_engine'] = self.trading_engine.get_engine_status()
        
        if self.portfolio_manager:
            state['portfolio'] = self.portfolio_manager.portfolio_state
        
        if self.risk_manager:
            state['risk_limits'] = self.risk_manager.risk_limits
        
        if self.market_regime_analyzer:
            state['market_regime'] = self.market_regime_analyzer.get_current_regime()
        
        return state
    
    def _print_position_dashboard(self):
        """
        Display real-time position dashboard with P&L.
        Phase 3.4 - Issue #105: Position Dashboard
        """
        try:
            if not self.portfolio_manager or not hasattr(self.portfolio_manager, 'get_open_positions'):
                return
            
            active_positions = self.portfolio_manager.get_open_positions() or {}
            
            if not active_positions:
                logger.info("\n📊 POSITION DASHBOARD: No open positions")
                return
            
            logger.info("\n" + "="*70)
            logger.info("📊 POSITION DASHBOARD")
            logger.info("="*70)
            
            total_unrealized_pnl = 0.0
            for position_id, position in active_positions.items():
                symbol = position.get('symbol', 'UNKNOWN')
                side = position.get('side', 'unknown')
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', entry_price)
                amount = position.get('amount', 0)
                
                # Calculate unrealized P&L
                unrealized_pnl = calculate_unrealized_pnl(side, entry_price, current_price, amount)
                
                pnl_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
                total_unrealized_pnl += unrealized_pnl
                
                # Format output
                pnl_symbol = "✅" if unrealized_pnl >= 0 else "❌"
                logger.info(f"{pnl_symbol} {symbol} {side.upper()}")
                logger.info(f"   Entry: ${entry_price:.4f} | Current: ${current_price:.4f}")
                logger.info(f"   Amount: {amount:.4f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)")
            
            logger.info("-"*70)
            total_symbol = "✅" if total_unrealized_pnl >= 0 else "❌"
            logger.info(f"{total_symbol} TOTAL UNREALIZED P&L: ${total_unrealized_pnl:.2f}")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Error displaying position dashboard: {e}")
    
    def _ohlcv_to_dataframe(self, ohlcv_data: List) -> pd.DataFrame:
        """Convert OHLCV list data to DataFrame."""
        import pandas as pd
        
        if not ohlcv_data:
            return None
        
        # OHLCV formatı: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Indicators ekle (eğer yoksa)
        from core.indicators import add_indicators
        if 'rsi' not in df.columns:
            df = add_indicators(df, self.config.get('indicators', {}))
        
        return df
    
    async def _fetch_ohlcv(self, client, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Helper method to fetch OHLCV data via REST API.
        
        This method runs the blocking client.ohlcv() call in a thread pool to prevent
        blocking the async event loop, which was causing the bot to freeze.
        
        Uses a 15-second timeout to prevent indefinite blocking on slow/failed requests.
        """
        try:
            # Run blocking I/O in thread pool with timeout to prevent indefinite blocking
            rows = await asyncio.wait_for(
                asyncio.to_thread(client.ohlcv, symbol, timeframe, limit=200),
                timeout=10.0
            )
            return self._ohlcv_to_dataframe(rows)
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout fetching {symbol} {timeframe} (10s limit)")
            return None
    
    async def _watchdog_loop(self):
        """
        Watchdog task that logs periodic heartbeats to detect loop stalls.
        
        Logs every 10 seconds regardless of main loop state to help diagnose
        if the main loop is truly stuck or just not logging.
        """
        logger.info("🐕 [WATCHDOG] Watchdog task started - will log every 60s")
        watchdog_count = 0
        
        try:
            while self.is_running:
                watchdog_count += 1
                logger.info(f"🐕 [WATCHDOG-{watchdog_count}] 💓 Heartbeat - is_running={self.is_running}")
                logger.info(f"   Active symbols: {len(self.active_symbols)}")
                logger.info(f"   Processed symbols: {self.processed_symbols_count}")
                
                # Check if engine is still running
                if self.trading_engine:
                    logger.info(f"   Engine state: {self.trading_engine.state.value}")
                
                # Force log flush
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.info("🐕 [WATCHDOG] Task cancelled")
        except Exception as e:
            logger.error(f"🐕 [WATCHDOG] Error: {e}", exc_info=True)
    
    async def _monitor_signal_queues(self):
        """
        Monitor signal queues and log their sizes periodically.
        
        Monitors:
        - StrategyCoordinator signal queue
        - LiveTradingEngine signal queue
        - Engine state
        - Execution statistics
        """
        logger.info("Queue monitoring task started")
        
        try:
            while self.is_running:
                try:
                    # Get queue sizes
                    coordinator_queue_size = 0
                    engine_queue_size = 0
                    engine_state = 'not_initialized'
                    
                    if self.strategy_coordinator:
                        coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
                    
                    if self.trading_engine:
                        engine_queue_size = self.trading_engine.signal_queue.qsize()
                        engine_state = self.trading_engine.state.value if hasattr(self.trading_engine, 'state') else 'unknown'
                    
                    # Log queue status
                    logger.info(f"📊 [QUEUE-MONITOR] Pipeline Status:")
                    logger.info(f"   StrategyCoordinator Queue: {coordinator_queue_size} signals")
                    logger.info(f"   LiveTradingEngine Queue: {engine_queue_size} signals")
                    logger.info(f"   LiveTradingEngine State: {engine_state}")
                    
                    # Get engine status if available
                    if self.trading_engine:
                        engine_status = self.trading_engine.get_engine_status()
                        logger.info(f"   Signals Received: {engine_status.get('signals_received', 0)}")
                        logger.info(f"   Signals Executed: {engine_status.get('signals_executed', 0)}")
                        logger.info(f"   Active Positions: {engine_status.get('active_positions', 0)}")
                    
                    # Alert if signals are stuck
                    if coordinator_queue_size > 5:
                        logger.warning(f"⚠️ [QUEUE-ALERT] {coordinator_queue_size} signals stuck in StrategyCoordinator queue!")
                        
                    if engine_queue_size > 5:
                        logger.warning(f"⚠️ [QUEUE-ALERT] {engine_queue_size} signals stuck in LiveTradingEngine queue!")
                        
                    if coordinator_queue_size > 0 and engine_queue_size == 0 and engine_state != 'running':
                        logger.critical(f"❌ [PIPELINE-BROKEN] Signals in coordinator but engine not running! State: {engine_state}")
                    
                    # Log lifecycle summary
                    if self.signal_lifecycle:
                        total_signals = len(self.signal_lifecycle)
                        stage_counts = {}
                        for signal_data in self.signal_lifecycle.values():
                            if signal_data['stages']:
                                last_stage = signal_data['stages'][-1]['stage']
                                stage_counts[last_stage] = stage_counts.get(last_stage, 0) + 1
                        
                        logger.info(f"📊 [LIFECYCLE] Total tracked: {total_signals} | Stages: {stage_counts}")
                    
                    # Wait 60 seconds before next check
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    logger.info("Queue monitoring task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in queue monitoring: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Fatal error in queue monitoring: {e}", exc_info=True)

    async def _monitor_rl_telemetry(self):
        """Periodically log RL telemetry stats for diagnostics."""
        logger.info("RL telemetry monitor task started")
        interval = self.config.get('monitoring', {}).get('rl_telemetry_interval_seconds', 300)
        interval = max(interval, 60)
        threshold = (
            self.config
            .get('ml', {})
            .get('reinforcement_learning', {})
            .get('q_std_bypass_threshold', 1e-4)
        )

        try:
            while self.is_running:
                await asyncio.sleep(interval)

                if not self.strategy_coordinator:
                    logger.debug("[RL-TELEMETRY] StrategyCoordinator not ready; skipping cycle")
                    continue

                try:
                    stats = self.strategy_coordinator.get_rl_telemetry_stats()
                except Exception as exc:
                    logger.error(f"[RL-TELEMETRY] Failed to collect stats: {exc}")
                    continue

                samples = stats.get('samples', 0)
                ppo_samples = stats.get('ppo_samples', 0)
                ppo_long_votes = stats.get('ppo_long_votes', 0)
                ppo_flat_votes = stats.get('ppo_flat_votes', 0)
                ppo_avg_score = stats.get('ppo_avg_score', 0.0)

                if not samples and not ppo_samples:
                    logger.info("📈 [RL-TELEMETRY] No RL decisions recorded yet")
                    continue

                if not samples and ppo_samples:
                    logger.info(
                        "📈 [RL-TELEMETRY] RL inactive | PPO samples=%s | avg_score=%.3f | long=%s | flat=%s",
                        ppo_samples,
                        ppo_avg_score,
                        ppo_long_votes,
                        ppo_flat_votes,
                    )
                    continue

                q_std_values = stats.get('q_std_values', [])
                sample_count = len(q_std_values) or samples
                q_std_med = stats.get('q_std_median', 0.0)
                q_range_med = stats.get('q_range_median', 0.0)
                veto_rate = stats.get('rl_veto_rate', 0.0) * 100
                bypass_rate = stats.get('rl_bypass_rate', stats.get('bypass_rate', 0.0)) * 100

                logger.info(
                    "📈 [RL-TELEMETRY] samples=%s | q_std_med=%.6f | q_range_med=%.6f | veto_rate=%.2f%% | bypass_rate=%.2f%% | PPO samples=%s | PPO avg=%.3f | PPO long=%s | PPO flat=%s",
                    sample_count,
                    q_std_med,
                    q_range_med,
                    veto_rate,
                    bypass_rate,
                    ppo_samples,
                    ppo_avg_score,
                    ppo_long_votes,
                    ppo_flat_votes,
                )

                if q_std_med < threshold:
                    logger.warning(
                        "📉 [RL-TELEMETRY] Median Q-std %.6f below threshold %.6f — RL model likely frozen",
                        q_std_med,
                        threshold
                    )

                if bypass_rate > 20.0:
                    logger.warning(
                        "⚠️ [RL-ALERT] High bypass rate (%.2f%%) - model may be frozen",
                        bypass_rate
                    )
        except asyncio.CancelledError:
            logger.info("RL telemetry monitor task cancelled")
        except Exception as exc:
            logger.error(f"RL telemetry monitor crashed: {exc}", exc_info=True)

    async def stop(self):
        """
        Graceful shutdown trigger for the coordinator.
        Delegates to the more specific stop_system method for compatibility.
        """
        logger.info("Coordinator 'stop()' called, delegating to 'stop_system()'.")
        if hasattr(self, 'stop_system') and callable(self.stop_system):
            await self.stop_system()
        else:
            logger.warning("'stop_system()' method not found on coordinator during shutdown.")
    
    async def stop_system(self):
        """
        Stop the production system gracefully.
        
        CRITICAL FIX: This method ONLY stops the trading loop and background tasks.
        It does NOT close positions or connections. The shutdown order is:
        1. Stop trading loop (this method) - prevents new signals
        2. Close positions (handled by launcher) - requires live connections
        3. Close WebSocket/exchange connections (handled by launcher)
        
        This ensures positions can be closed successfully before connections die.
        """
        logger.info("Stopping production system...")
        self.is_running = False
        
        # Cancel monitoring task
        if hasattr(self, '_monitoring_task') and self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel watchdog task
        if hasattr(self, '_watchdog_task') and self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

        # Cancel RL telemetry task
        if hasattr(self, '_rl_telemetry_task') and self._rl_telemetry_task:
            self._rl_telemetry_task.cancel()
            try:
                await self._rl_telemetry_task
            except asyncio.CancelledError:
                pass

        coordinator = getattr(self, "strategy_coordinator", None)
        if coordinator and hasattr(coordinator, "stop_fast_watcher"):
            try:
                coordinator.stop_fast_watcher()
            except Exception:
                pass
        if coordinator and hasattr(coordinator, "stop_micro_gate_watcher"):
            try:
                coordinator.stop_micro_gate_watcher()
            except Exception:
                pass
        
        # Stop trading engine (but keep connections alive)
        if self.trading_engine:
            await self.trading_engine.stop_live_trading()
        
        # CRITICAL FIX: WebSocket/exchange connections are NOT closed here
        # They must remain open until positions are closed by the launcher's cleanup()
        # The launcher will close connections in the correct order after positions are closed
        
        logger.info("Production system stopped (connections remain open for position closure)")
