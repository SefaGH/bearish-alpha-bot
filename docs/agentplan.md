## **1. Signal Integrity / Price-Deviation Guard**

### **Repo-Gerçeği: Sinyal modeli dict (Pydantic değil)**
Bu repo’nun core loop’u `TradingSignal(BaseModel)` kullanmıyor; sinyaller `dict` olarak taşınıyor.

MVP hedefi: her stratejinin, sinyal içine en azından `meta.price_meta` koymasını standartlaştırmak.
`src/strategies/mean_reversion.py` zaten bunu yapıyor.

```python
# src/strategies/* (örnek şablon)
signal.setdefault("meta", {}).setdefault(
    "price_meta",
    {
        # referans alınan fiyat (genelde entry/limit)
        "price_used": float(signal.get("entry") or 0.0) or None,
        "price_used_source": "strategy_calculated",

        # sinyal üretildiği anda gözlenen piyasa fiyatı
        "market_price": float(market_price) if market_price is not None else None,
        "market_price_source": str(market_price_source) if market_price_source else None,
        "market_price_fallback_chain": list(market_price_fallback_chain) if market_price_fallback_chain else None,
    },
)
```

### **IntegrityGuard Implementation:**
```python
# src/safety/signal_integrity_guard.py
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime, timezone

from src.core.signal_intents import INTENT_CLOSE


class SignalIntegrityGuard:
    """Sinyal bütünlük kontrolü ve price deviation guard"""
    
    def __init__(self, config: Dict, market_data_pipeline):
        self.config = config.get('signals', {}).get('integrity_guard', {})
        self.market_data_pipeline = market_data_pipeline
        self.enabled = self.config.get('enabled', True)
        
    async def validate(self, signal: Dict[str, Any], current_position: Any = None) -> Dict[str, Any]:
        """
        Sinyali doğrula ve aksiyon belirle
        
        Returns:
            Dict with keys: valid(bool), action(str), reason(str), metadata(dict)
        """
        if not self.enabled:
            return {'valid': True, 'action': 'pass', 'reason': 'disabled'}
        
        # 1. Staleness kontrolü (signal/meta timestampleri repo’ya göre opsiyonel)
        staleness_result = self._check_staleness(signal)
        if not staleness_result['valid']:
            return staleness_result
        
        # 2. Price deviation kontrolü
        deviation_result = await self._check_price_deviation(signal)
        if not deviation_result['valid']:
            # Intent'e göre aksiyon belirle
            return self._determine_action_based_on_intent(
                signal, 
                deviation_result, 
                current_position
            )
        
        return {'valid': True, 'action': 'pass', 'reason': 'valid'}
    
    def _check_staleness(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Sinyalin bayat olup olmadığını kontrol et"""
        max_staleness_ms = self.config.get('max_staleness_ms', 10000)  # 10 saniye
        
        # Candle timestamp'inden kontrol
        meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
        price_meta = meta.get("price_meta") if isinstance(meta.get("price_meta"), dict) else {}
        candle_ts = price_meta.get('candle_close_ts')
        if candle_ts:
            # Eğer datetime ise UTC varsay; değilse staleness kontrolünü skip etmek güvenli
            try:
                now = datetime.now(timezone.utc)
                if isinstance(candle_ts, datetime) and candle_ts.tzinfo is None:
                    candle_ts = candle_ts.replace(tzinfo=timezone.utc)
                staleness_ms = (now - candle_ts).total_seconds() * 1000
            except Exception:
                staleness_ms = 0
            if staleness_ms > max_staleness_ms:
                return {
                    'valid': False,
                    'action': 'reject',
                    'reason': f'stale_candle_{staleness_ms:.0f}ms',
                    'metadata': {'staleness_ms': staleness_ms}
                }
        
        # Signal timestamp'inden kontrol
        signal_ts = signal.get("timestamp")
        if signal_ts is not None:
            try:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                ts_ms = int(float(signal_ts))
                signal_age_ms = max(0, now_ms - ts_ms)
                if signal_age_ms > max_staleness_ms:
                    return {
                        'valid': False,
                        'action': 'reject',
                        'reason': f'stale_signal_{signal_age_ms:.0f}ms',
                        'metadata': {'signal_age_ms': signal_age_ms}
                    }
            except Exception:
                pass
        
        return {'valid': True}
    
    async def _check_price_deviation(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Fiyat sapmasını kontrol et"""
        max_deviation_pct = self.config.get('max_deviation_pct', 0.001)  # %0.1
        
        # Reference price'ı belirle
        meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
        price_meta = meta.get("price_meta") if isinstance(meta.get("price_meta"), dict) else {}
        reference_price = price_meta.get('price_used') or signal.get("entry")
        if not reference_price:
            return {'valid': True, 'reason': 'no_reference_price'}
        
        # Current price'ı al (WS-first) — SSOT: MarketDataPipeline.get_latest_price
        symbol = signal.get("symbol")
        tf = signal.get("timeframe") or "1m"
        current_price = await self.market_data_pipeline.get_latest_price(str(symbol), timeframe=str(tf))
        
        if not current_price:
            return {
                'valid': False,
                'action': 'reject',
                'reason': 'integrity_data_unavailable',
            }
        
        # Sapmayı hesapla
        deviation_pct = abs(current_price - reference_price) / reference_price
        
        if deviation_pct > max_deviation_pct:
            return {
                'valid': False,
                'action': 'deviation_detected',
                'reason': f'price_deviation_{deviation_pct:.4f}',
                'metadata': {
                    'deviation_pct': deviation_pct,
                    'reference_price': reference_price,
                    'current_price': current_price,
                    'max_allowed': max_deviation_pct
                }
            }
        
        return {'valid': True}
    
    def _determine_action_based_on_intent(self, signal: Dict[str, Any], 
                                         deviation_result: Dict,
                                         current_position: Any = None) -> Dict:
        """Intent'e göre aksiyon belirle"""
        # Signal intent'ini belirle (henüz set edilmemiş olabilir)
        intent = self._infer_intent(signal, current_position)
        
        if intent in ['entry', 'reentry', 'scale_in']:
            return {
                'valid': False,
                'action': 'reject',
                'reason': f'integrity_price_deviation_{intent}',
                'metadata': deviation_result['metadata']
            }
        
        elif intent == 'reverse':
            # Reverse'u INTENT_CLOSE'a çevir (close-only davranış)
            return {
                'valid': True,
                'action': 'convert_reverse_to_close',
                'reason': 'integrity_reverse_to_close',
                'metadata': {
                    **deviation_result['metadata'],
                    'original_intent': 'reverse',
                    'new_intent': INTENT_CLOSE,
                }
            }
        
        return deviation_result
    
    def _infer_intent(self, signal: Dict[str, Any], position: Any) -> str:
        """Mevcut duruma göre intent çıkar"""
        if not position:
            return 'entry'
        
        signal_side = str(signal.get("side") or "").strip().lower()
        position_side = getattr(position, "side", None)
        if position_side is None and isinstance(position, dict):
            position_side = position.get("side")
        position_side = str(position_side or "").strip().lower()

        if signal_side and position_side and signal_side == position_side:
            # NOTE: _should_scale_in aşağıda minimal stub olarak tanımlı; repo’daki mevcut scale-in
            # kurallarıyla entegre edilmelidir.
            return 'scale_in' if self._should_scale_in(position, signal) else 'exit_same_side'
        
        return 'reverse'

    def _should_scale_in(self, position: Any, signal: Dict[str, Any]) -> bool:
        """
        Minimal stub (plan netleştirme amaçlı): repo’daki scale-in kurallarıyla değiştir.
        Varsayılan: False (scale-in yerine exit_same_side).
        """
        return False
```

### **StrategyCoordinator Entegrasyonu:**
```python
# src/core/strategy_coordinator.py içinde sinyal pipeline (repo-native)
# ÖNEMLİ: IntegrityGuard, `_enrich_signal(...)` SONRASI ve auto-reverse tagging ÖNCESİ çalışmalı.
from typing import Any, Dict

from src.core.signal_intents import MAINTENANCE_INTENTS, INTENT_CLOSE


async def _process_signal(self, signal: dict) -> None:
    enriched = await self._enrich_signal(signal)

    integrity = await self.integrity_guard.validate(enriched, current_position=self._get_current_position(enriched.get("symbol")))
    if integrity.get("action") == "reject":
        self._log_integrity_rejection(enriched, integrity)
        return

    if integrity.get("action") == "convert_reverse_to_close":
        enriched["intent"] = INTENT_CLOSE
        enriched.setdefault("meta", {}).setdefault("integrity", {})
        enriched["meta"]["integrity"].update(
            {
                "status": "converted",
                "reason": integrity.get("reason"),
                "deviation_pct": (integrity.get("metadata") or {}).get("deviation_pct"),
                "original_intent": "reverse",
            }
        )

    regime = await self.regime_filter.validate(enriched)
    if not regime.get("valid", True):
        self._log_regime_rejection(enriched, regime)
        return

    # Auto-reverse tagging, maintenance intent'leri override etmemeli.
    if str(enriched.get("intent") or "").lower() not in {i.lower() for i in MAINTENANCE_INTENTS}:
        self._tag_auto_reverse(enriched)

    self._log_integrity_guard_result(enriched, integrity)
```

## **2. Regime Filter**

### **RegimeDetector Implementation:**
```python
# src/safety/regime_detector.py
import logging
from datetime import datetime
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Piyasa rejimini tespit eder ve strateji filtrelemesi yapar"""
    
    def __init__(self, config: Dict, market_data_pipeline):
        self.config = config.get('signals', {}).get('regime_filter', {})
        self.market_data_pipeline = market_data_pipeline
        self.enabled = self.config.get('enabled', True)
        
        # Strateji-rejim mapping
        self.strategy_regime_mapping = self.config.get('strategy_regime_mapping', {
            'mean_reversion': ['range'],
            'adaptive_ob': ['range', 'transitional', 'crash_rebound'],
            'adaptive_str': ['trend', 'transitional']
        })
        
        # Rejim konfigürasyonu
        self.regime_config = {
            # Repo-gerçeği: DI+/DI- yok; temel eşikler ADX üzerinden.
            'trend': {'min_adx': 25},
            'range': {'max_adx': 20},
            'transitional': {'min_adx': 20, 'max_adx': 25},
            'crash_rebound': {'min_price_drop_pct': 0.03, 'max_recovery_pct': 0.01}
        }
    
    async def detect_regime(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Piyasa rejimini tespit et"""
        try:
            # Repo-gerçeği:
            # - MarketDataPipeline.get_latest_ohlcv(...) -> DataFrame + indikatör kolonları lowercase
            # - `adx`, `atr`, `ema50` ve alias `ema_mid` mevcut
            # - DI+/DI- kolonları yok
            df = await self.market_data_pipeline.get_latest_ohlcv(symbol, timeframe=timeframe, limit=250)
            if df is None or getattr(df, "empty", True):
                return {
                    'label': 'unknown',
                    'confidence': 0.0,
                    'reason': 'no_ohlcv',
                    'timeframe_used': timeframe,
                    'timestamp': datetime.utcnow(),
                }

            def _last_float(col: str):
                try:
                    if col in df.columns:
                        return float(df[col].iloc[-1])
                except Exception:
                    return None
                return None

            close_px = _last_float('close')
            adx = _last_float('adx')
            atr = _last_float('atr')
            ema_mid = _last_float('ema_mid')
            if ema_mid is None:
                ema_mid = _last_float('ema50')

            atr_pct = (atr / close_px) if (atr and close_px) else 0.0
            trend_direction = None
            if close_px is not None and ema_mid is not None:
                trend_direction = 'up' if close_px >= ema_mid else 'down'
            
            # Rejim belirleme
            regime = {
                'adx': adx,
                'atr_pct': atr_pct,
                'timeframe_used': timeframe,
                'trend_direction': trend_direction,
                'timestamp': datetime.utcnow()
            }
            
            # Rejim sınıflandırma
            if adx is not None and adx > 25:
                regime['label'] = 'trend'
                regime['confidence'] = min(adx / 50, 1.0)  # Normalize
                
            elif adx is not None and adx < 20:
                regime['label'] = 'range'
                regime['confidence'] = 1.0 - (adx / 20)
                
            elif self._is_crash_rebound(symbol):
                regime['label'] = 'crash_rebound'
                regime['confidence'] = 0.8
                
            else:
                regime['label'] = 'transitional'
                regime['confidence'] = 0.5
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection failed for {symbol}: {e}")
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }

    def _is_crash_rebound(self, symbol: str) -> bool:
        """
        Minimal stub (plan netleştirme amaçlı): gerçek crash/rebound algısı repo’daki
        mevcut price-drop/mean-reversion heuristikleriyle değiştirilmeli.
        Varsayılan: False.
        """
        return False
    
    def is_strategy_allowed(self, strategy_name: str, regime: Dict) -> Tuple[bool, str, float]:
        """Bu strateji şu anki rejimde çalışabilir mi?"""
        if not self.enabled:
            return True, "disabled", 1.0
        
        regime_label = regime.get('label', 'unknown')
        allowed_regimes = self.strategy_regime_mapping.get(strategy_name, [])
        
        # Hard veto: strateji bu rejimde kesinlikle yasak
        if allowed_regimes and regime_label not in allowed_regimes:
            return False, f"strategy_not_allowed_in_{regime_label}", 0.0
        
        # Soft penalty: rejim confidence düşükse ağırlık azalt
        regime_confidence = regime.get('confidence', 0.5)
        min_confidence = self.config.get('min_regime_confidence', 0.3)
        
        if regime_confidence < min_confidence:
            # Düşük güven: weight penalty uygula
            penalty = regime_confidence / min_confidence
            return True, f"low_regime_confidence_{regime_confidence:.2f}", penalty
        
        return True, f"allowed_in_{regime_label}", 1.0
```

### **RegimeFilter Entegrasyonu:**
```python
# src/safety/regime_filter.py
from typing import Any, Dict

from src.safety.regime_detector import MarketRegimeDetector

class RegimeFilter:
    """Stratejileri rejime göre filtreler"""
    
    def __init__(self, config: Dict, market_data_pipeline):
        self.detector = MarketRegimeDetector(config, market_data_pipeline)
        self.config = config.get('signals', {}).get('regime_filter', {})
        self.enabled = self.config.get('enabled', True)
    
    async def validate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Sinyali rejim filtresinden geçir"""
        if not self.enabled:
            return {'valid': True, 'action': 'pass', 'reason': 'disabled'}

        strategy = signal.get('strategy_name') or signal.get('strategy')
        symbol = signal.get('symbol')
        tf = signal.get('timeframe') or '1h'
        if not strategy or not symbol:
            return {'valid': True, 'action': 'pass', 'reason': 'missing_strategy_or_symbol'}

        # Rejimi tespit et
        regime = await self.detector.detect_regime(str(symbol), timeframe=str(tf))

        # Strateji izin kontrolü
        allowed, reason, weight = self.detector.is_strategy_allowed(str(strategy), regime)
        
        if not allowed:
            return {
                'valid': False,
                'action': 'reject',
                'reason': f'regime_veto_{reason}',
                'metadata': {
                    'regime': regime,
                    'strategy': strategy
                }
            }
        
        # Rejim bilgisini sinyale ekle (repo-native)
        meta = signal.get('meta') if isinstance(signal.get('meta'), dict) else {}
        meta['regime'] = regime
        signal['meta'] = meta

        # Weight `regime_weight` olarak core pipeline’da zaten tüketilebiliyor
        signal['regime_weight'] = float(weight)
        
        # Weight threshold kontrolü
        min_weight = self.config.get('min_regime_weight', 0.3)
        if weight < min_weight:
            return {
                'valid': False,
                'action': 'reject',
                'reason': f'low_regime_weight_{weight:.2f}',
                'metadata': {
                    'regime': regime,
                    'weight': weight,
                    'min_required': min_weight
                }
            }
        
        return {
            'valid': True,
            'action': 'pass',
            'reason': f'allowed_in_{regime.get("label", "unknown")}',
            'metadata': {'regime': regime, 'weight': weight}
        }
```

## **3. Position Stickiness / Transition Policy**

### **TransitionPolicy Implementation:**
```python
# core/transition_policy.py
from typing import Any, Dict


class PositionTransitionPolicy:
    """Cross-strategy reversal politikalarını uygular"""
    
    def __init__(self, config: Dict):
        self.config = config.get('signals', {}).get('transition_policy', {})
        self.enabled = self.config.get('enabled', True)
        
        # Strateji aileleri
        self.strategy_families = {
            'adaptive_ob': 'trend_following',
            'adaptive_str': 'trend_following',
            'mean_reversion': 'mean_reversion'
        }
        
        # Default politika matrisi
        self.policy_matrix = self._build_policy_matrix()
    
    def _build_policy_matrix(self) -> Dict:
        """Politika matrisini oluştur"""
        return {
            # (from_family, to_family, direction, intent) -> action
            ('trend_following', 'mean_reversion', 'opposite', 'reverse'): {
                'action': 'convert_to_close',
                'reason': 'cross_strategy_trend_to_counter',
                'allow_force': False
            },
            ('mean_reversion', 'trend_following', 'opposite', 'reverse'): {
                'action': 'allow',  # Counter-to-trend izin ver
                'reason': 'counter_to_trend_allowed',
                'min_profit_pct': 0.5
            },
            # Same family içinde reverse izin ver
            ('trend_following', 'trend_following', 'opposite', 'reverse'): {
                'action': 'allow',
                'reason': 'same_family_reverse'
            },
            ('mean_reversion', 'mean_reversion', 'opposite', 'reverse'): {
                'action': 'allow',
                'reason': 'same_family_reverse'
            },
            # Same direction (scale-in)
            ('*', '*', 'same', 'entry'): {
                'action': 'scale_in_check',
                'reason': 'same_direction_scale_in'
            }
        }
    
    def evaluate(self, current_position: Any, 
                 incoming_signal: Dict[str, Any],
                 inferred_intent: str) -> Dict:
        """
        Geçiş politikasını değerlendir
        
        Returns:
            Dict with: allowed(bool), action(str), reason(str), metadata(dict)
        """
        if not self.enabled or not current_position:
            return {
                'allowed': True,
                'action': 'allow',
                'reason': 'no_position_or_disabled'
            }
        
        # Parametreleri hazırla
        from_family = self.strategy_families.get(current_position.strategy, 'unknown')
        to_family = self.strategy_families.get(incoming_signal.get('strategy_name') or incoming_signal.get('strategy'), 'unknown')
        
        incoming_side = str(incoming_signal.get('side') or '').strip().lower()
        direction = 'same' if incoming_side == current_position.side else 'opposite'
        
        # Force reverse bayrağını kontrol et
        meta = incoming_signal.get('meta') if isinstance(incoming_signal.get('meta'), dict) else {}
        if meta.get('force_reverse_allowed', False):
            return {
                'allowed': True,
                'action': 'allow',
                'reason': 'force_reverse_flagged'
            }
        
        # Politika matrisinde ara
        policy_key = (from_family, to_family, direction, inferred_intent)
        policy = self.policy_matrix.get(policy_key)
        
        # Wildcard fallback
        if not policy:
            wildcard_key = ('*', '*', direction, inferred_intent)
            policy = self.policy_matrix.get(wildcard_key)
        
        if not policy:
            # Bilinmeyen politika - default: izin ver
            return {
                'allowed': True,
                'action': 'allow',
                'reason': 'no_policy_found_default_allow'
            }
        
        # Politika aksiyonunu uygula
        action = policy['action']
        
        if action == 'allow':
            # Ek kontroller (min profit gibi)
            if 'min_profit_pct' in policy:
                if current_position.unrealized_pnl_pct < policy['min_profit_pct']:
                    return {
                        'allowed': False,
                        'action': 'convert_to_close',
                        'reason': f'insufficient_profit_{current_position.unrealized_pnl_pct:.2f}%',
                        'metadata': {'required': policy['min_profit_pct']}
                    }
            
            return {
                'allowed': True,
                'action': 'allow',
                'reason': policy['reason']
            }
        
        elif action == 'convert_to_close':
            return {
                'allowed': False,  # Reverse'a izin yok
                'action': 'convert_to_close',
                'reason': policy['reason'],
                'metadata': {
                    'from_family': from_family,
                    'to_family': to_family,
                    'original_intent': inferred_intent
                }
            }
        
        elif action == 'reject':
            return {
                'allowed': False,
                'action': 'reject',
                'reason': policy['reason']
            }
        
        # Default: izin ver
        return {
            'allowed': True,
            'action': 'allow',
            'reason': 'policy_default'
        }
```

### **StrategyCoordinator'da Entegrasyon:**
```python
# core/strategy_coordinator.py - _tag_auto_reverse metodunda
import logging
from typing import Any, Dict

from src.core.signal_intents import INTENT_CLOSE, INTENT_REVERSE

logger = logging.getLogger(__name__)


def _tag_auto_reverse(self, signal: Dict[str, Any]) -> None:
    """Auto-reverse tag'le - GÜNCELLENMİŞ"""
    current_position = self._get_current_position(signal.get('symbol'))
    
    signal_side = str(signal.get('side') or '').strip().lower()
    if not current_position or signal_side == current_position.side:
        return
    
    # Transition policy kontrolü
    policy_result = self.transition_policy.evaluate(
        current_position,
        signal,
        inferred_intent='reverse'
    )
    
    if not policy_result['allowed']:
        if policy_result['action'] == 'convert_to_close':
            # Reverse yerine INTENT_CLOSE (close-only davranış)
            signal['intent'] = INTENT_CLOSE
            signal.setdefault('meta', {}).setdefault('transition_policy', {})
            signal['meta']['transition_policy'] = {
                'original_intent': 'reverse',
                'action': 'converted',
                'reason': policy_result['reason']
            }
            logger.info(
                "Cross-strategy reverse blocked, converted to close | %s -> %s | reason=%s",
                current_position.strategy,
                signal.get('strategy_name') or signal.get('strategy'),
                policy_result.get('reason'),
            )
            return
        elif policy_result['action'] == 'reject':
            # Sinyali tamamen reddet (pipeline'da drop et)
            signal.setdefault('meta', {}).setdefault('drop', {})
            signal['meta']['drop'] = {
                'kind': 'transition_policy',
                'reason': policy_result['reason'],
            }
            logger.info(
                "Cross-strategy transition rejected | %s -> %s | reason=%s",
                current_position.strategy,
                signal.get('strategy_name') or signal.get('strategy'),
                policy_result.get('reason'),
            )
            return
    
    # Politika izin veriyor - reverse tag'le
    signal['intent'] = INTENT_REVERSE
    signal.setdefault('meta', {})
    signal['meta']['reverse_from_position_id'] = current_position.id
    signal['meta']['transition_policy'] = {
        'action': 'allowed',
        'reason': policy_result['reason']
    }
```

### **Conflict Resolution Entegrasyonu:**
```python
# core/strategy_coordinator.py - conflict resolution'da
from typing import Dict, Any, List, Optional
from src.core.signal_intents import INTENT_CLOSE


def _resolve_signal_conflict(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Sinyal çakışmasını çöz - GÜNCELLENMİŞ"""
    winner = self._select_winning_signal(signals)
    
    if not winner:
        return None
    
    # Eğer winner reverse olacaksa transition policy kontrol et
    current_position = self._get_current_position(winner.get('symbol'))
    winner_side = str(winner.get('side') or '').strip().lower()
    if current_position and winner_side != current_position.side:
        policy_result = self.transition_policy.evaluate(
            current_position,
            winner,
            inferred_intent='reverse'
        )
        
        if not policy_result['allowed'] and policy_result['action'] == 'convert_to_close':
            # Conflict resolution winner'ı INTENT_CLOSE yap
            winner['intent'] = INTENT_CLOSE
            winner.setdefault('meta', {}).setdefault('conflict_resolution', {})
            winner['meta']['conflict_resolution'] = {
                'original_intent': 'reverse',
                'action': 'converted',
                'reason': policy_result['reason']
            }
    
    return winner
```

## **LiveTradingEngine'de close (close-only davranış) İşleme:**
```python
# Not: Repo'da yeni bir close_only intent eklemeye gerek yok.
# Close-only davranış için mevcut INTENT_CLOSE kullanılır.
# LiveTradingEngine/PositionManager zaten "close" intent'ini destekler.
```

## **Telemetri ve Loglama:**
```python
# core/telemetry.py
import json
from datetime import datetime
from typing import Any, Dict


class SignalTelemetry:
    """Sinyal telemetrisi"""
    
    def log_integrity_guard(self, signal: Dict[str, Any], result: Dict) -> None:
        meta = signal.get('meta') if isinstance(signal.get('meta'), dict) else {}
        integrity_meta = meta.get('integrity') if isinstance(meta.get('integrity'), dict) else {}
        structured_log = {
            'event': 'integrity_guard',
            'timestamp': datetime.utcnow().isoformat(),
            'signal_id': signal.get('signal_id'),
            'symbol': signal.get('symbol'),
            'strategy': signal.get('strategy_name') or signal.get('strategy'),
            'action': result.get('action'),
            'reason': result.get('reason'),
            'valid': result.get('valid'),
            'metadata': result.get('metadata', {}),
            'deviation_pct': integrity_meta.get('deviation_pct'),
            'integrity_status': integrity_meta.get('status')
        }
        self.logger.info(json.dumps(structured_log, default=str))
    
    def log_regime_filter(self, signal: Dict[str, Any], result: Dict) -> None:
        meta = signal.get('meta') if isinstance(signal.get('meta'), dict) else {}
        structured_log = {
            'event': 'regime_filter',
            'timestamp': datetime.utcnow().isoformat(),
            'signal_id': signal.get('signal_id'),
            'symbol': signal.get('symbol'),
            'strategy': signal.get('strategy_name') or signal.get('strategy'),
            'regime': (meta.get('regime') or {}).get('label') if isinstance(meta.get('regime'), dict) else None,
            'regime_confidence': (meta.get('regime') or {}).get('confidence') if isinstance(meta.get('regime'), dict) else None,
            'regime_weight': signal.get('regime_weight'),
            'action': result.get('action'),
            'reason': result.get('reason'),
            'valid': result.get('valid')
        }
        self.logger.info(json.dumps(structured_log, default=str))
    
    def log_transition_policy(self, signal: Dict[str, Any], position: Any, result: Dict) -> None:
        structured_log = {
            'event': 'transition_policy',
            'timestamp': datetime.utcnow().isoformat(),
            'signal_id': signal.get('signal_id'),
            'symbol': signal.get('symbol'),
            'current_strategy': getattr(position, 'strategy', None) if position else None,
            'incoming_strategy': signal.get('strategy_name') or signal.get('strategy'),
            'direction': 'same' if position and str(signal.get('side') or '').strip().lower() == getattr(position, 'side', None) else 'opposite',
            'intent': signal.get('intent'),
            'action': result.get('action'),
            'reason': result.get('reason'),
            'allowed': result.get('allowed'),
            'metadata': result.get('metadata', {})
        }
        self.logger.info(json.dumps(structured_log, default=str))
```

## **Konfigürasyon:**
```yaml
signals:
  integrity_guard:
    enabled: true
    max_staleness_ms: 10000  # 10 saniye
    max_deviation_pct: 0.001  # %0.1
    
  regime_filter:
    enabled: true
    min_regime_confidence: 0.3
    min_regime_weight: 0.3
    strategy_regime_mapping:
      mean_reversion: ["range"]
      adaptive_ob: ["range", "transitional", "crash_rebound"]
      adaptive_str: ["trend", "transitional"]
      
  transition_policy:
    enabled: true
    allow_cross_strategy_reverse: false
        default_action: "convert_to_close"
    
strategies:
  mean_reversion:
    price_meta_enabled: true
    price_source: "closed_candle"  # veya "forming_candle"
    
  adaptive_ob:
    price_meta_enabled: true
    price_source: "mid"
    
  adaptive_str:
    price_meta_enabled: true
    price_source: "mid"
```

## **Test Planı:**
1. **Unit Testler**: Her guard için ayrı testler
2. **Integration Test**: StrategyCoordinator akış testi
3. **Backtest**: Tarihsel verilerle whipsaw azalımı testi
4. **Canlı Dry-Run**: 24 saat log analizi ile false positive/negative oranları

Bu implementasyonla:
1. **Bayat mum sorunu** çözülür (Integrity Guard)
2. **Rejim çatışmaları** azalır (Regime Filter)  
3. **Cross-strategy reversal** engellenir (Transition Policy)
4. **Telemetri** ile tüm kararlar izlenebilir
5. **Config-driven** esneklik sağlanır


### **MİKRO-OPTIMİZASYON VE GÜVENLİK EKLEMELERİ**

#### **1. Integrity Guard: Fail-Safe Mekanizması (Güvenli Başarısızlık)**

Mevcut kodda:

```python
if not current_price:
    return {'valid': True, 'reason': 'no_current_price'}  # RİSKLİ! (fail-open)

```

* **Risk:** Eğer WebSocket koparsa veya fiyat çekilemezse, Integrity Guard "Pas" geçiyor. Yani en kör olduğumuz anda botu denetimsiz bırakıyoruz.
* **Öneri:** Veri yoksa işlem yapmak kumardır. Default davranış **REJECT** olmalı.
* **Düzeltme:**
```python
if not current_price:
    return {
        'valid': False, 
        'action': 'reject', 
        'reason': 'integrity_data_unavailable'
    }

```



#### **2. Regime Filter: Hesaplama Önbelleği (Caching)**

* **Durum:** `StrategyCoordinator` her sinyal işlediğinde `detect_regime` çağırıyor. Eğer 5 strateji aynı saniyede sinyal üretirse, ADX/ATR hesaplaması 5 kez tekrar edilir.
* **Öneri:** Rejim tespiti her saniye değişmez. Basit bir "Time-to-Live" (TTL) cache ekle.
* **Düzeltme (`MarketRegimeDetector` içine):**
```python
from datetime import datetime

async def detect_regime(self, symbol, timeframe):
    if not hasattr(self, "_cache"):
        self._cache = {}
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in self._cache:
        last_update, data = self._cache[cache_key]
        if (datetime.utcnow() - last_update).total_seconds() < 60: # 1 dk Cache
            return data

    # ... hesaplama ...
    self._cache[cache_key] = (datetime.utcnow(), regime)
    return regime

```



#### **3. Dinamik Sapma Eşiği (ATR Entegrasyonu)**

Konuştuğumuz üzere, sabit `%0.1` (`0.001`) sapma, düşük volatilitede çok geniş, yüksek volatilitede (haber anında) çok dar kalabilir.

* **Düzeltme (`_check_price_deviation` içine):**
```python
# Config'den statik değer al
base_max_deviation = self.config.get('max_deviation_pct', 0.001)

# ATR varsa dinamik eşik hesapla (ATR'nin %10'u kadar esneklik tanı)
meta = signal.get("meta") if isinstance(signal.get("meta"), dict) else {}
price_meta = meta.get("price_meta") if isinstance(meta.get("price_meta"), dict) else {}
atr_pct = price_meta.get('strategy_specific', {}).get('atr_pct')
if atr_pct:
    dynamic_max = max(base_max_deviation, atr_pct * 0.1) 
    max_allowed = dynamic_max
else:
    max_allowed = base_max_deviation

```

---

## **Appendix: Soft Deferral / Incubator (Repo-Native Snippets)**

### **ÇOK ÖNEMLİ: `STRATEGY_RECHECK` semantiği (Replay değil)**
`refresh_policy="STRATEGY_RECHECK"` seçersen, incubator **orijinal sinyali replay etmez**.
Due olduğunda incubator item’ı **drop eder** ve bir `strategy_recheck_request` event’i üretir.

Bu şu anlama gelir:
- “Aynı sinyali biraz sonra tekrar sıraya koy” hedefin varsa `STRATEGY_RECHECK` doğru araç değildir.
- `STRATEGY_RECHECK` sadece “stratejiye tekrar hesap yapmasını söyle” mekanizmasıdır; bunu tüketen bir recheck worker/consumer olmalıdır.

### **A) Aynı sinyali daha sonra replay etmek (inkübasyon)**
Repo’daki gerçek imza:
- `await StrategyCoordinator.incubate_signal(strategy_name, signal, reason_code, refresh_policy, stage=None)`
- `wait_time` parametresi yok; gecikme/TTL policy’den gelir.

```python
# src/core/strategy_coordinator.py (içeriden)

result = await self.incubate_signal(
    strategy_name=str(signal.get("strategy_name") or ""),
    signal=dict(signal),
    reason_code="strategy.soft_deferral",
    refresh_policy="REPRICE_AND_RESIZE",  # replay sırasında fiyat/size refresh et
    stage="integrity_guard",
)
return result
```

### **B) Soft deferral event ile (schema validasyonu + inkübasyon)**
`handle_soft_deferral(...)` beklediği event schema (minimum):
- `event_type="soft_deferral_event"`
- `strategy`/`strategy_name`, `symbol`, `side`, `timeframe`, `setup_anchor_ts_ms`
- opsiyonel: `reason_code`, `reason`, `condition_data`, `refresh_policy`

```python
# src/core/strategy_coordinator.py (örnek kullanım)

event = {
    "event_type": "soft_deferral_event",
    "strategy": "adaptive_ob",
    "symbol": "BTC/USDT",
    "side": "long",
    "timeframe": "5m",
    "setup_anchor_ts_ms": int(time.time() * 1000),
    "reason_code": "strategy.soft_deferral",
    "reason": "integrity_guard: price deviated; wait for recheck",

    # Opsiyonel: FAST_PRICE_WATCH için trigger_price vb.
    "condition_data": {
        # "trigger_price": 42000.0,
        # "ttl_ms": 60000,
        # "max_checks": 20,
    },

    # Eğer set edilmezse:
    # - trigger_price varsa FAST_PRICE_WATCH
    # - yoksa STRATEGY_RECHECK
    "refresh_policy": "REPRICE_AND_RESIZE",
}

result = await self.handle_soft_deferral(event)
```


---

## **Implementation Checklist / DoD (Eksik Bir Şey Kalmasın)**

Bu dokümandaki snippet’leri birebir uygulamadan önce aşağıdaki maddelerin **tamamı** net olmalı:

1. **`STRATEGY_RECHECK` operasyonel koşulu**
    - `refresh_policy="STRATEGY_RECHECK"` **sadece** `strategy_recheck_request` event’lerini tüketen bir **recheck worker/consumer** mevcutsa kullanılmalı.
    - Böyle bir consumer yoksa `STRATEGY_RECHECK` seçmek, incubator item’ının due olduğunda **drop edilmesi** ve “bot sinyali kaçırdı” gibi görünmesiyle sonuçlanır.
    - “Aynı sinyali biraz sonra tekrar değerlendirmek/replay etmek” hedefi için `refresh_policy="REPRICE_AND_RESIZE"` (veya repo’daki replay policy’lerden uygun olanı) kullanılmalı.

2. **Fail-safe davranışı (data yoksa trade yok)**
    - `IntegrityGuard` fiyat verisi bulamazsa (`get_latest_price` None/invalid) **reject** etmeli (`integrity_data_unavailable`).
    - Regime tespiti için OHLCV yoksa `regime_weight` **konservatif** olmalı (örn. 0.0) veya filter PASS etmemeli.

3. **Repo-native API uyumu**
    - Sinyal tipi: `dict` (attribute access yok).
    - Fiyat/ohlcv kaynağı: `MarketDataPipeline.get_latest_price(...)` ve `get_latest_ohlcv(...)`.
    - İndikatör kolonları: `adx`, `atr`, `ema50` ve/veya `ema_mid` (DI kolonlarına güvenilmez).
    - Close-only davranışı: yeni intent icat etmeden `INTENT_CLOSE` kullanılır.

4. **Incubator/soft deferral doğrulaması**
    - `incubate_signal(...)` imzasında `wait_time` yok; gecikme/TTL policy’den gelir.
    - Soft deferral event schema alanları (strategy/symbol/side/timeframe/setup_anchor_ts_ms) tam ve tipleri doğru.
    - `reason_code` değerleri log/metric tarafında kolay filtrelenebilir (örn. `integrity_guard.price_deviation`).

5. **Gözlemlenebilirlik (minimum)**
    - Reject/deferral kararları `reason_code` + kritik metriklerle loglanır (`symbol`, `side`, `timeframe`, `deviation_pct`, `regime_weight`).
    - `STRATEGY_RECHECK` seçildiğinde loglarda consumer’ın çalıştığı net görülebilir (recheck request üretildi / işlendi).

6. **Test kapsamı (en azından)**
    - Integrity: “fiyat yoksa reject” ve “sapma büyükse reject/deferral” senaryoları.
    - Regime: “OHLCV yoksa konservatif ağırlık” ve “trend/range ayrımı” senaryoları.
    - TransitionPolicy: “convert_to_close -> INTENT_CLOSE” davranışı ve reverse whipsaw azaltma senaryosu.

7. **Implementasyon durumu (bu doküman = plan)**
   - Bu dokümanda `RegimeDetector/RegimeFilter` ve `TransitionPolicy` için verilen kod blokları **repo’ya uyumlu taslak/snippet** niteliğindedir; bu bileşenler şu an core’da “hazır özellik” olarak var sayılmamalı.
   - “Uygulamaya hazır” ifadesi, planın repo API/semantiğiyle çelişmemesi anlamındadır; gerçek üretim davranışı için bu bileşenlerin koda alınması ve en azından minimal unit/integration testlerle doğrulanması gerekir.



---

### **TEMSİLİ AKIŞ DİYAGRAMI (GÖRSEL TEYİT)**

Kodlamaya başlarken zihninde şu akışı canlandırman için süreci görselleştirelim. Bu bölüm **temsili bir akıştır** (yüksek seviye); gerçek kodda adımlar bazı yerlerde iç içe geçebilir ama karar noktaları aynıdır.

**Akış Özeti:**

1. **Ham Sinyal Gelir:** (`MeanReversion`, `Price: 78.100`)
2. **Enrich:** Yan veriler eklenir.
3. **Integrity Guard:**
* *Soru:* Şu anki fiyat 78.100'e yakın mı?
* *Cevap:* Hayır, fiyat 78.300 oldu (Fark > %0.1). -> **REJECT** veya **SOFT DEFERRAL (incubator: REPRICE_AND_RESIZE)**


4. **Regime Filter (Eğer geçerse):**
* *Soru:* Piyasa Yatay mı?
* *Cevap:* Evet (ADX < 20). MeanReversion için uygun. -> **PASS (regime_weight: 1.0)**
* *Not:* `regime_weight` threshold altına düşerse -> **REJECT** (trade yok)


5. **Transition Policy (Reverse niyeti varsa):**
* *Soru:* Elimde Trend Long var, bu sinyal Short. Reverse edeyim mi?
* *Cevap:* Hayır, Trend'i bozma. -> **CONVERT TO CLOSE** (signal intent: `INTENT_CLOSE`, close-only davranış)


6. **Conflict Resolution:** Diğer sinyallerle yarışır (winner seçilir; gerekiyorsa winner da `INTENT_CLOSE`’a dönüştürülebilir).
7. **Engine:** `close` intentini görür, Long'u kapatır ama Short açmaz.

---

### **SONUÇ**

Hazırladığın taslak **uygulamaya hazır bir implementasyon planıdır**. En iyi sonucu almak için, konfigürasyon eşiklerini (deviation/TTL/regime thresholds) testlerle ve replay/dry-run ile kademeli olarak ayarlaman gerekir.