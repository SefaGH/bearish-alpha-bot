"""
Test Suite for GEMMA Dynamic Feature Adaptation System

Tests for:
1. Manifest loading and validation
2. Dynamic feature extraction with different modes
3. Regime predictor with manifest dimensions
4. RL agent with manifest state size
5. End-to-end pipeline with dynamic features
6. Legacy compatibility
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.ml.manifest_manager import ManifestManager
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.price_predictor import AdvancedPricePredictionEngine


class DummyFeaturePipeline:
    """Minimal feature pipeline that mirrors manifest-driven expectations."""

    def __init__(self, bundle_path: str, feature_names):
        self.models_config = {'active_bundle': bundle_path}
        self._feature_names = feature_names

    def extract_features(self, price_data, mode: str = 'price'):
        latest_close = float(price_data['close'].iloc[-1]) if price_data is not None else 0.0
        row = {feature: latest_close for feature in self._feature_names}
        return pd.DataFrame([row])


class DummyMarketDataPipeline:
    """Returns a fixed OHLCV frame to drive the prediction engine."""

    def __init__(self, price_frame: pd.DataFrame):
        self._frame = price_frame

    async def get_latest_ohlcv(self, symbol: str, timeframe: str, exchange=None):  # noqa: D401
        return self._frame.copy()


@pytest.fixture
def sample_price_frame():
    """Create deterministic OHLCV data for price engine tests."""
    dates = pd.date_range('2024-01-01', periods=120, freq='15min')
    base = np.linspace(100, 105, len(dates))
    noise = np.random.randn(len(dates)) * 0.05
    close = base + noise
    frame = pd.DataFrame({
        'open': close - 0.1,
        'high': close + 0.2,
        'low': close - 0.2,
        'close': close,
        'volume': np.random.randint(1_000, 2_000, len(dates))
    }, index=dates)
    return frame


@pytest.fixture
def gemma_test_bundle(tmp_path):
    """Create a temporary GEMMA bundle with manifest and stub files."""
    bundle = tmp_path / 'gemma_bundle'
    bundle.mkdir()

    feature_names = [f'feature_{i}' for i in range(8)]
    manifest = {
        'version': 'test-2.0.0',
        'mode': 'gemma',
        'feature_count': len(feature_names),
        'feature_names_ordered': feature_names,
        'selected_features_price': list(range(len(feature_names))),
        'selected_features_regime': list(range(len(feature_names))),
        'price_model_path': 'gemma_price_model.pt',
        'price_scaler_path': 'gemma_price_scaler.joblib',
        'active_features_path': 'price_features.json',
        'rl_state_size': len(feature_names)
    }

    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    (bundle / 'gemma_price_model.pt').write_text('stub-model')
    (bundle / 'gemma_price_scaler.joblib').write_text('stub-scaler')
    (bundle / 'price_features.json').write_text(json.dumps({'features': feature_names}))

    return {
        'path': str(bundle),
        'manifest': manifest,
        'feature_names': feature_names
    }


@pytest.fixture
def patched_gemma_adapter(monkeypatch):
    """Patch GemmaTorchScriptAdapter with a lightweight test double."""

    class _DummyAdapter:
        def __init__(self, config):
            self.config = config
            self.model = object()

        def predict(self, feature_snapshot):
            return {
                'probabilities': [0.25, 0.25, 0.5],
                'price_confidence': 0.83,
                'prediction': 2,
                'prediction_label': 'bullish'
            }

    monkeypatch.setattr('src.ml.price_predictor.GemmaTorchScriptAdapter', _DummyAdapter)
    return _DummyAdapter


class TestManifestManager:
    """Test ManifestManager functionality"""
    
    def test_manifest_manager_singleton(self):
        """ManifestManager should be a singleton"""
        mgr1 = ManifestManager()
        mgr2 = ManifestManager()
        assert mgr1 is mgr2, "ManifestManager should be a singleton"
    
    def test_load_legacy_manifest(self):
        """Should load legacy manifest successfully"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        assert manifest is not None
        assert manifest['feature_count'] == 42
        assert manifest['mode'] == 'legacy'
        assert len(manifest['feature_names_ordered']) == 42
    
    def test_get_selected_features_price(self):
        """Should return correct price features from manifest"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        price_features = mgr.get_selected_features('price')
        assert len(price_features) == 42
        assert isinstance(price_features, list)
    
    def test_get_selected_features_regime(self):
        """Should return correct regime features from manifest"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        regime_features = mgr.get_selected_features('regime')
        assert len(regime_features) == 42
        assert isinstance(regime_features, list)
    
    def test_manifest_validation(self):
        """Manifest should validate required fields"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        # Check required fields
        assert 'feature_count' in manifest
        assert 'feature_names_ordered' in manifest
        assert 'selected_features_price' in manifest
        assert 'selected_features_regime' in manifest
        
        # Validate consistency
        assert len(manifest['feature_names_ordered']) == manifest['feature_count']


class TestFeatureEngineeringPipeline:
    """Test dynamic feature extraction"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data
    
    def test_feature_pipeline_initialization_with_manifest(self):
        """Feature pipeline should initialize with manifest"""
        config = {
            'models': {'active_bundle': 'artifacts/legacy'}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        
        assert hasattr(pipeline, 'manifest')
        assert hasattr(pipeline, 'expected_feature_count')
        assert pipeline.expected_feature_count == 42
    
    def test_extract_features_default_mode(self, sample_data):
        """Should extract features in default price mode"""
        config = {
            'models': {'active_bundle': 'artifacts/legacy'}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        features = pipeline.extract_features(sample_data)
        
        assert not features.empty
        # Should return features (may be less than 42 due to NaN handling)
        assert len(features.columns) > 0
    
    def test_extract_features_price_mode(self, sample_data):
        """Should extract price features when mode='price'"""
        config = {
            'models': {'active_bundle': 'artifacts/legacy'}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        features = pipeline.extract_features(sample_data, mode='price')
        
        assert not features.empty
        # With legacy manifest, should attempt to return price features
        assert len(features.columns) > 0
    
    def test_extract_features_regime_mode(self, sample_data):
        """Should extract regime features when mode='regime'"""
        config = {
            'models': {'active_bundle': 'artifacts/legacy'}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        features = pipeline.extract_features(sample_data, mode='regime')
        
        assert not features.empty
        # With legacy manifest, should attempt to return regime features
        assert len(features.columns) > 0
    
    def test_feature_pipeline_fallback_without_manifest(self, sample_data):
        """Should fall back gracefully if manifest loading fails"""
        config = {
            'models': {'active_bundle': 'nonexistent/path'}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        
        # Should have fallen back to defaults
        assert hasattr(pipeline, 'expected_feature_count')
        # Default fallback is 42
        assert pipeline.expected_feature_count == 42


class TestDynamicDimensions:
    """Test dynamic dimension handling across components"""
    
    @pytest.fixture
    def test_manifest(self, tmp_path):
        """Create a test manifest with different feature count"""
        manifest = {
            "version": "test-1.0",
            "mode": "test",
            "feature_count": 50,
            "feature_names_ordered": [f"feature_{i}" for i in range(50)],
            "selected_features_price": list(range(50)),
            "selected_features_regime": list(range(50)),
            "rl_state_size": 50,
            "regime_scaler_path": "scaler.pkl",
            "regime_model_path": "model.pth",
            "metadata": {
                "description": "Test manifest with 50 features"
            }
        }
        
        bundle_path = tmp_path / "test_bundle"
        bundle_path.mkdir()
        
        with open(bundle_path / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        return str(bundle_path)
    
    def test_feature_pipeline_with_custom_manifest(self, test_manifest):
        """Feature pipeline should adapt to custom manifest"""
        config = {
            'models': {'active_bundle': test_manifest}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        
        assert pipeline.expected_feature_count == 50
        assert len(pipeline.price_features) == 50
        assert len(pipeline.regime_features) == 50


class TestLegacyCompatibility:
    """Test backward compatibility with legacy 42-feature system"""
    
    def test_legacy_manifest_loads(self):
        """Legacy manifest should load successfully"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        assert manifest['mode'] == 'legacy'
        assert manifest['feature_count'] == 42
    
    def test_legacy_feature_names_present(self):
        """Legacy manifest should have expected feature names"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        feature_names = manifest['feature_names_ordered']
        
        # Check some expected legacy feature names
        assert 'technical_rsi' in feature_names
        assert 'technical_macd' in feature_names
        assert 'microstructure_volume' in feature_names
        assert 'volatility_vol_5' in feature_names
        assert 'momentum_roc_5' in feature_names
    
    def test_legacy_rl_state_size(self):
        """Legacy manifest should specify correct RL state size"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        assert manifest['rl_state_size'] == 42


class TestManifestConsistency:
    """Test manifest internal consistency"""
    
    def test_feature_count_matches_names(self):
        """Feature count should match number of feature names"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        assert len(manifest['feature_names_ordered']) == manifest['feature_count']
    
    def test_selected_features_within_bounds(self):
        """Selected feature indices should be within valid range"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('artifacts/legacy')
        
        feature_count = manifest['feature_count']
        
        # Check price features
        for idx in manifest['selected_features_price']:
            assert 0 <= idx < feature_count, f"Price feature index {idx} out of bounds"
        
        # Check regime features
        for idx in manifest['selected_features_regime']:
            assert 0 <= idx < feature_count, f"Regime feature index {idx} out of bounds"


class TestErrorHandling:
    """Test error handling and fallback behavior"""
    
    def test_missing_manifest_fallback(self):
        """Should fall back to defaults when manifest is missing"""
        mgr = ManifestManager()
        manifest = mgr.load_manifest('nonexistent/path')
        
        # Should have created default manifest
        assert manifest is not None
        assert 'feature_count' in manifest
        # Default should be 42
        assert manifest['feature_count'] == 42
    
    def test_feature_pipeline_handles_missing_manifest(self):
        """Feature pipeline should handle missing manifest gracefully"""
        config = {
            'models': {'active_bundle': 'nonexistent/path'}
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        
        # Should initialize with defaults
        assert pipeline.expected_feature_count == 42


class TestAdvancedPricePredictionEngineIntegration:
    """Validate manifest-driven price engine behavior."""

    def test_engine_initializes_with_manifest_bundle(
        self,
        gemma_test_bundle,
        sample_price_frame,
        patched_gemma_adapter,
    ):
        """Engine should load manifest metadata and activate GEMMA adapter."""

        feature_pipeline = DummyFeaturePipeline(gemma_test_bundle['path'], gemma_test_bundle['feature_names'])
        market_pipeline = DummyMarketDataPipeline(sample_price_frame)

        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=market_pipeline,
            feature_pipeline=feature_pipeline,
            config={'active_bundle': gemma_test_bundle['path'], 'timeframes': ['15m']},
        )

        assert engine.is_trained is True
        assert engine.manifest['version'] == gemma_test_bundle['manifest']['version']
        assert engine.bundle_path == Path(gemma_test_bundle['path'])
        assert 'ML Mode' in engine.get_status_summary()

    @pytest.mark.asyncio
    async def test_engine_generates_manifest_prediction(
        self,
        gemma_test_bundle,
        sample_price_frame,
        patched_gemma_adapter,
    ):
        """Engine should produce GEMMA-mode predictions using manifest features."""

        feature_pipeline = DummyFeaturePipeline(gemma_test_bundle['path'], gemma_test_bundle['feature_names'])
        market_pipeline = DummyMarketDataPipeline(sample_price_frame)

        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=market_pipeline,
            feature_pipeline=feature_pipeline,
            config={
                'active_bundle': gemma_test_bundle['path'],
                'timeframes': ['15m'],
                'classification_to_pct_scale': 2.0,
            },
        )

        forecast = await engine._make_prediction_for_symbol('BTC/USDT')

        assert forecast is not None
        assert forecast['mode'] == 'gemma'
        assert 'aggregated' in forecast
        assert forecast['aggregated']['forecast'].shape[0] == 1

    def test_engine_enters_fallback_when_bundle_missing(self, tmp_path):
        """Missing bundles should keep the engine in fallback mode."""

        missing_bundle = tmp_path / 'missing_bundle'
        feature_pipeline = DummyFeaturePipeline(str(missing_bundle), ['feature_0'])

        engine = AdvancedPricePredictionEngine(
            market_data_pipeline=None,
            feature_pipeline=feature_pipeline,
            config={'active_bundle': str(missing_bundle)},
        )

        assert engine.is_trained is False
        assert 'FALLBACK Mode' in engine.get_status_summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
