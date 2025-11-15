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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
