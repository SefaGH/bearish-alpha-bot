"""
Test import compatibility for both package and script execution contexts.

This test validates that the dual import strategy works correctly
for all four core modules that use pnl_calculator utilities.
"""

import pytest
import sys
import os
from pathlib import Path


class TestPackageImports:
    """Test imports work in package context (python -m, from src.*)."""
    
    def test_risk_manager_package_import(self):
        """Test risk_manager can be imported as a package."""
        import src.core.risk_manager
        assert hasattr(src.core.risk_manager, 'RiskManager')
    
    def test_position_manager_package_import(self):
        """Test position_manager can be imported as a package."""
        import src.core.position_manager
        assert hasattr(src.core.position_manager, 'AdvancedPositionManager')
    
    def test_realtime_risk_package_import(self):
        """Test realtime_risk can be imported as a package."""
        import src.core.realtime_risk
        assert hasattr(src.core.realtime_risk, 'RealTimeRiskMonitor')
    
    def test_production_coordinator_package_import(self):
        """Test production_coordinator can be imported as a package."""
        # This might fail due to missing dependencies, but import should not raise ImportError
        try:
            import src.core.production_coordinator
            assert hasattr(src.core.production_coordinator, 'ProductionCoordinator')
        except ModuleNotFoundError as e:
            # Allow missing dependencies like ccxt, but not import structure issues
            if 'pnl_calculator' in str(e):
                pytest.fail(f"pnl_calculator import failed: {e}")
            # Otherwise it's an expected dependency issue, skip
            pytest.skip(f"Missing optional dependency: {e}")


class TestScriptStyleImports:
    """Test imports work when src/ is in PYTHONPATH (script execution style)."""
    
    def setup_method(self):
        """Add src to path like scripts do."""
        repo_root = Path(__file__).parent.parent
        src_path = repo_root / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            self._added_to_path = True
        else:
            self._added_to_path = False
    
    def teardown_method(self):
        """Clean up path modifications."""
        if self._added_to_path:
            repo_root = Path(__file__).parent.parent
            src_path = str(repo_root / 'src')
            if src_path in sys.path:
                sys.path.remove(src_path)
    
    def test_pnl_calculator_direct_import(self):
        """Test pnl_calculator can be imported directly."""
        from utils.pnl_calculator import calculate_unrealized_pnl
        result = calculate_unrealized_pnl('long', 100, 110, 1)
        assert result == 10.0
    
    def test_risk_manager_with_src_in_path(self):
        """Test risk_manager imports pnl_calculator correctly when src is in path."""
        # Force reload to test import in current path context
        import importlib
        import core.risk_manager
        importlib.reload(core.risk_manager)
        assert hasattr(core.risk_manager, 'RiskManager')
    
    def test_position_manager_with_src_in_path(self):
        """Test position_manager imports pnl_calculator correctly when src is in path."""
        import importlib
        import core.position_manager
        importlib.reload(core.position_manager)
        assert hasattr(core.position_manager, 'AdvancedPositionManager')


class TestDualImportFunctionality:
    """Test that both import paths lead to the same functionality."""
    
    def test_both_import_styles_work(self):
        """Test that both import styles work and produce same results."""
        # Package-style import
        from src.utils.pnl_calculator import calculate_unrealized_pnl as calc1
        
        # Add src to path for script-style import
        repo_root = Path(__file__).parent.parent
        src_path = str(repo_root / 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Script-style import
        from utils.pnl_calculator import calculate_unrealized_pnl as calc2
        
        # Both should work identically (they may be different objects due to import caching)
        assert calc1('long', 100, 110, 1) == calc2('long', 100, 110, 1)
        assert calc1('short', 100, 90, 1) == calc2('short', 100, 90, 1)
    
    def test_pnl_functions_accessible_from_core_modules(self):
        """Test that core modules can successfully use pnl_calculator functions."""
        import src.core.risk_manager as rm
        import src.core.position_manager as pm
        
        # These modules should have successfully imported the functions
        # We verify by checking the module loads without ImportError
        assert rm.RiskManager is not None
        assert pm.AdvancedPositionManager is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
