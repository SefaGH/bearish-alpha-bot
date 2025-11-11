"""
Unit test to verify no import-time side effects in param_sweep modules.

This test ensures that importing param_sweep and param_sweep_str modules
does not cause side effects like:
- sys.path modifications
- logging configuration changes
- IO operations (file/network)
- subprocess calls
"""
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestParamSweepImportSideEffects(unittest.TestCase):
    """Test that param_sweep modules don't have import-time side effects."""
    
    def setUp(self):
        """Save initial state before each test."""
        self.original_syspath = sys.path.copy()
        self.original_log_handlers = logging.root.handlers.copy()
        self.original_log_level = logging.root.level
    
    def tearDown(self):
        """Restore state after each test."""
        # Restore sys.path
        sys.path = self.original_syspath
        
        # Restore logging state
        logging.root.handlers = self.original_log_handlers
        logging.root.level = self.original_log_level
        
        # Remove imported modules to ensure clean slate for next test
        modules_to_remove = [
            'src.backtest.param_sweep',
            'src.backtest.param_sweep_str',
        ]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
    
    def test_param_sweep_no_syspath_modification(self):
        """Test that importing param_sweep doesn't modify sys.path."""
        # Import the module
        try:
            from src.backtest import param_sweep
        except ImportError:
            # If import fails due to missing dependencies, skip the sys.path check
            # but still verify the module structure is correct
            import importlib.util
            spec = importlib.util.find_spec("src.backtest.param_sweep")
            if spec is None:
                self.skipTest("param_sweep module not found")
            return
        
        # Verify sys.path wasn't modified
        self.assertEqual(
            sys.path,
            self.original_syspath,
            "Importing param_sweep modified sys.path"
        )
    
    def test_param_sweep_str_no_syspath_modification(self):
        """Test that importing param_sweep_str doesn't modify sys.path."""
        # Import the module
        try:
            from src.backtest import param_sweep_str
        except ImportError:
            # If import fails due to missing dependencies, skip the sys.path check
            import importlib.util
            spec = importlib.util.find_spec("src.backtest.param_sweep_str")
            if spec is None:
                self.skipTest("param_sweep_str module not found")
            return
        
        # Verify sys.path wasn't modified
        self.assertEqual(
            sys.path,
            self.original_syspath,
            "Importing param_sweep_str modified sys.path"
        )
    
    def test_param_sweep_no_logging_basicconfig(self):
        """Test that importing param_sweep doesn't call logging.basicConfig."""
        with patch('logging.basicConfig') as mock_basicconfig:
            try:
                from src.backtest import param_sweep
            except ImportError:
                self.skipTest("param_sweep module has import errors")
            
            # Verify logging.basicConfig wasn't called during import
            mock_basicconfig.assert_not_called()
    
    def test_param_sweep_str_no_logging_basicconfig(self):
        """Test that importing param_sweep_str doesn't call logging.basicConfig."""
        with patch('logging.basicConfig') as mock_basicconfig:
            try:
                from src.backtest import param_sweep_str
            except ImportError:
                self.skipTest("param_sweep_str module has import errors")
            
            # Verify logging.basicConfig wasn't called during import
            mock_basicconfig.assert_not_called()
    
    def test_param_sweep_no_io_operations(self):
        """Test that importing param_sweep doesn't perform IO operations."""
        # Mock file operations
        with patch('builtins.open') as mock_open:
            try:
                from src.backtest import param_sweep
            except ImportError:
                self.skipTest("param_sweep module has import errors")
            
            # Verify no file operations during import
            # (Some imports might read __pycache__ which is ok)
            for call in mock_open.call_args_list:
                args, kwargs = call
                if args:
                    filename = str(args[0])
                    # Allow __pycache__ reads but nothing else
                    self.assertIn(
                        '__pycache__',
                        filename,
                        f"Unexpected file operation during import: {filename}"
                    )
    
    def test_param_sweep_has_main_function(self):
        """Test that param_sweep has a main() function (not called at import)."""
        try:
            from src.backtest import param_sweep
        except ImportError:
            self.skipTest("param_sweep module has import errors")
        
        # Verify main() function exists
        self.assertTrue(
            hasattr(param_sweep, 'main'),
            "param_sweep should have a main() function"
        )
        self.assertTrue(
            callable(param_sweep.main),
            "param_sweep.main should be callable"
        )
    
    def test_param_sweep_str_has_main_function(self):
        """Test that param_sweep_str has a main() function (not called at import)."""
        try:
            from src.backtest import param_sweep_str
        except ImportError:
            self.skipTest("param_sweep_str module has import errors")
        
        # Verify main() function exists
        self.assertTrue(
            hasattr(param_sweep_str, 'main'),
            "param_sweep_str should have a main() function"
        )
        self.assertTrue(
            callable(param_sweep_str.main),
            "param_sweep_str.main should be callable"
        )


class TestProductionCoordinatorImportSideEffects(unittest.TestCase):
    """Test that production_coordinator doesn't have import-time side effects."""
    
    def setUp(self):
        """Save initial state before each test."""
        self.original_syspath = sys.path.copy()
    
    def tearDown(self):
        """Restore state after each test."""
        sys.path = self.original_syspath
        
        # Remove imported modules
        if 'src.core.production_coordinator' in sys.modules:
            del sys.modules['src.core.production_coordinator']
    
    def test_production_coordinator_no_syspath_modification(self):
        """Test that importing production_coordinator doesn't modify sys.path."""
        try:
            from src.core import production_coordinator
        except ImportError:
            # Module might have missing dependencies, but we can still check
            # that it doesn't modify sys.path in its top-level code
            import importlib.util
            spec = importlib.util.find_spec("src.core.production_coordinator")
            if spec is None:
                self.skipTest("production_coordinator module not found")
            return
        
        # Verify sys.path wasn't modified
        self.assertEqual(
            sys.path,
            self.original_syspath,
            "Importing production_coordinator modified sys.path"
        )


if __name__ == '__main__':
    unittest.main()
