# src/ml/manifest_manager.py
import threading
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ManifestManager:
    """Thread-safe singleton manifest manager"""
    
    _instance = None
    _lock = threading.Lock()
    _manifest = None
    _feature_name_to_idx = {}
    _bundle_path = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, bundle_path: str = None):
        """Initialize with bundle path (thread-safe)"""
        with self._lock:
            if not self._initialized or bundle_path != self._bundle_path:
                self._bundle_path = bundle_path
                self._manifest = None  # Force reload
                self._feature_name_to_idx = {}
                self._initialized = True
                logger.info(f"ManifestManager initialized with bundle: {bundle_path}")
    
    def load_manifest(self, bundle_path: str = None) -> Dict[str, Any]:
        """Load and cache manifest (thread-safe)"""
        with self._lock:
            # Use provided path or cached path
            if bundle_path:
                self._bundle_path = bundle_path
            
            # Return cached if available and same path
            if self._manifest is not None and bundle_path is None:
                return self._manifest
            
            # Resolve symlink if exists
            if self._bundle_path:
                path = Path(self._bundle_path)
                if path.is_symlink():
                    actual_path = path.resolve()
                    logger.info(f"Resolved symlink {path} → {actual_path}")
                    path = actual_path
                
                manifest_path = path / "manifest.json"
            else:
                # Fallback to legacy
                manifest_path = Path("artifacts/legacy/manifest.json")
            
            if not manifest_path.exists():
                logger.warning(f"Manifest not found at {manifest_path}, using defaults")
                self._manifest = self._create_default_manifest()
            else:
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
                    logger.info(f"✅ Loaded manifest: {self._manifest.get('version', 'unknown')}")
                    logger.info(f"   Feature count: {self._manifest['feature_count']}")
            
            # Validate and build mappings
            self._validate_and_index()
            
            return self._manifest
    
    def _validate_and_index(self):
        """Validate manifest and build feature mappings"""
        if not self._manifest:
            return
            
        # Required fields
        required = ["feature_count", "feature_names_ordered"]
        for field in required:
            if field not in self._manifest:
                raise ValueError(f"Manifest missing required field: {field}")
        
        # Build feature name to index mapping
        feature_names = self._manifest.get("feature_names_ordered", [])
        self._feature_name_to_idx = {
            name: idx for idx, name in enumerate(feature_names)
        }
        
        # Validate consistency
        if len(feature_names) != self._manifest["feature_count"]:
            raise ValueError(
                f"Feature count mismatch: {len(feature_names)} names != "
                f"{self._manifest['feature_count']} count"
            )
    
    def get_feature_indices(self, feature_names: List[str]) -> List[int]:
        """Convert feature names to indices (thread-safe)"""
        with self._lock:
            if not self._feature_name_to_idx:
                raise ValueError("Feature mappings not initialized")
            
            indices = []
            for name in feature_names:
                if name not in self._feature_name_to_idx:
                    raise ValueError(f"Unknown feature: {name}")
                indices.append(self._feature_name_to_idx[name])
            
            return indices
    
    def get_selected_features(self, mode: str = "price") -> List[str]:
        """Get selected feature names for mode (thread-safe)"""
        with self._lock:
            if not self._manifest:
                self.load_manifest()
            
            # Get indices
            if mode == "price":
                indices = self._manifest.get("selected_features_price", [])
            elif mode == "regime":
                indices = self._manifest.get("selected_features_regime", [])
            else:
                # Return all features
                return self._manifest.get("feature_names_ordered", [])
            
            # Convert indices to names
            all_features = self._manifest.get("feature_names_ordered", [])
            return [all_features[i] for i in indices if i < len(all_features)]
    
    def get_model_path(self, model_type: str) -> Path:
        """Get absolute model path (thread-safe)"""
        with self._lock:
            if not self._manifest:
                self.load_manifest()
            
            # Get relative path from manifest
            path_key = f"{model_type}_model_path"
            relative_path = self._manifest.get(path_key)
            
            if not relative_path:
                raise ValueError(f"Model path not found for: {model_type}")
            
            # Make absolute
            if self._bundle_path:
                base = Path(self._bundle_path)
            else:
                base = Path("artifacts/legacy")
            
            return base / relative_path
    
    def _create_default_manifest(self) -> Dict[str, Any]:
        """Create default manifest for fallback"""
        return {
            "version": "0.0-default",
            "mode": "legacy",
            "feature_count": 42,
            "feature_names_ordered": [f"feature_{i}" for i in range(42)],
            "selected_features_price": list(range(42)),
            "selected_features_regime": list(range(42)),
            "rl_state_size": 42,
            "metadata": {
                "warning": "Using default manifest - no bundle found"
            }
        }
