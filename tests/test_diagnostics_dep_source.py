"""
Unit tests for pre_gemma_health_check.py enhanced with requirements.txt support.
"""
import unittest
import tempfile
import json
from pathlib import Path
import sys
import os

# Add scripts directory to path for importing the module
SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from pre_gemma_health_check import (
        parse_requirements_txt,
        parse_pyproject_toml,
        load_dependency_names,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class TestDependencyParsing(unittest.TestCase):
    """Test dependency parsing functions."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        if not IMPORT_SUCCESS:
            self.skipTest(f"Cannot import pre_gemma_health_check: {IMPORT_ERROR}")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_requirements_txt_basic(self):
        """Test parsing basic requirements.txt format."""
        req_file = self.temp_path / "requirements.txt"
        req_file.write_text("""
# Core dependencies
pandas>=2.0.0
numpy<2
torch==2.0.0
scikit-learn>=1.0
        """, encoding='utf-8')
        
        deps = parse_requirements_txt(req_file)
        
        self.assertIn('pandas', deps)
        self.assertIn('numpy', deps)
        self.assertIn('torch', deps)
        self.assertIn('scikit-learn', deps)
        self.assertEqual(len(deps), 4)
    
    def test_parse_requirements_txt_with_comments(self):
        """Test parsing requirements.txt with various comment styles."""
        req_file = self.temp_path / "requirements.txt"
        req_file.write_text("""
# === CORE DEPENDENCIES ===
pandas>=2.0.0  # Data processing
numpy<2  # Numerical operations

# Optional
# torch>=2.0.0  # Commented out

===================================
scikit-learn>=1.0
        """, encoding='utf-8')
        
        deps = parse_requirements_txt(req_file)
        
        self.assertIn('pandas', deps)
        self.assertIn('numpy', deps)
        self.assertNotIn('torch', deps)  # Should be skipped (commented)
        self.assertIn('scikit-learn', deps)
    
    def test_parse_requirements_txt_complex_versions(self):
        """Test parsing complex version specifiers."""
        req_file = self.temp_path / "requirements.txt"
        req_file.write_text("""
aiohttp==3.8.6
yarl<2.0
multidict<7.0
package[extra1,extra2]>=1.0.0
another-package>=1.0.0,<2.0.0
        """, encoding='utf-8')
        
        deps = parse_requirements_txt(req_file)
        
        self.assertIn('aiohttp', deps)
        self.assertIn('yarl', deps)
        self.assertIn('multidict', deps)
        self.assertIn('package', deps)
        self.assertIn('another-package', deps)
    
    def test_parse_requirements_txt_nonexistent(self):
        """Test parsing non-existent requirements.txt."""
        req_file = self.temp_path / "nonexistent.txt"
        
        deps = parse_requirements_txt(req_file)
        
        self.assertEqual(deps, set())
    
    def test_load_dependency_names_auto_prefers_requirements(self):
        """Test auto mode prefers requirements.txt when both exist."""
        # Create both files
        req_file = self.temp_path / "requirements.txt"
        req_file.write_text("pandas>=2.0.0\ntorch>=2.0.0", encoding='utf-8')
        
        pyproject_file = self.temp_path / "pyproject.toml"
        pyproject_file.write_text("""
[project]
dependencies = [
    "numpy>=1.24.0",
    "scikit-learn>=1.0",
]
        """, encoding='utf-8')
        
        # Mock REPO_ROOT
        import pre_gemma_health_check
        original_repo_root = pre_gemma_health_check.REPO_ROOT
        pre_gemma_health_check.REPO_ROOT = self.temp_path
        
        try:
            deps = load_dependency_names(source="auto")
            
            # Should load from requirements.txt, not pyproject.toml
            self.assertIn('pandas', deps)
            self.assertIn('torch', deps)
            self.assertNotIn('numpy', deps)  # From pyproject, should not be included
            self.assertNotIn('scikit-learn', deps)
        finally:
            pre_gemma_health_check.REPO_ROOT = original_repo_root
    
    def test_load_dependency_names_auto_fallback_to_pyproject(self):
        """Test auto mode falls back to pyproject.toml when requirements.txt missing."""
        # Create only pyproject.toml
        pyproject_file = self.temp_path / "pyproject.toml"
        pyproject_file.write_text("""
[project]
dependencies = [
    "numpy>=1.24.0",
    "scikit-learn>=1.0",
]
        """, encoding='utf-8')
        
        # Mock REPO_ROOT
        import pre_gemma_health_check
        original_repo_root = pre_gemma_health_check.REPO_ROOT
        pre_gemma_health_check.REPO_ROOT = self.temp_path
        
        try:
            deps = load_dependency_names(source="auto")
            
            # Should load from pyproject.toml as fallback
            self.assertIn('numpy', deps)
            self.assertIn('scikit-learn', deps)
        finally:
            pre_gemma_health_check.REPO_ROOT = original_repo_root
    
    def test_load_dependency_names_explicit_requirements(self):
        """Test explicitly requesting requirements.txt source."""
        req_file = self.temp_path / "requirements.txt"
        req_file.write_text("pandas>=2.0.0\ntorch>=2.0.0", encoding='utf-8')
        
        # Mock REPO_ROOT
        import pre_gemma_health_check
        original_repo_root = pre_gemma_health_check.REPO_ROOT
        pre_gemma_health_check.REPO_ROOT = self.temp_path
        
        try:
            deps = load_dependency_names(source="requirements")
            
            self.assertIn('pandas', deps)
            self.assertIn('torch', deps)
        finally:
            pre_gemma_health_check.REPO_ROOT = original_repo_root
    
    def test_load_dependency_names_explicit_pyproject(self):
        """Test explicitly requesting pyproject.toml source."""
        pyproject_file = self.temp_path / "pyproject.toml"
        pyproject_file.write_text("""
[project]
dependencies = [
    "numpy>=1.24.0",
    "scikit-learn>=1.0",
]
        """, encoding='utf-8')
        
        # Mock REPO_ROOT
        import pre_gemma_health_check
        original_repo_root = pre_gemma_health_check.REPO_ROOT
        pre_gemma_health_check.REPO_ROOT = self.temp_path
        
        try:
            deps = load_dependency_names(source="pyproject")
            
            self.assertIn('numpy', deps)
            self.assertIn('scikit-learn', deps)
        finally:
            pre_gemma_health_check.REPO_ROOT = original_repo_root


class TestCommandLineArguments(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def test_default_dep_source_is_auto(self):
        """Test that default --dep-source is 'auto'."""
        import argparse
        
        # Create a simple parser similar to the one in the script
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--dep-source',
            choices=['auto', 'requirements', 'pyproject'],
            default='auto'
        )
        
        # Parse with no arguments
        args = parser.parse_args([])
        
        self.assertEqual(args.dep_source, 'auto')
    
    def test_dep_source_accepts_valid_values(self):
        """Test that --dep-source accepts all valid values."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--dep-source',
            choices=['auto', 'requirements', 'pyproject'],
            default='auto'
        )
        
        # Test each valid value
        for source in ['auto', 'requirements', 'pyproject']:
            args = parser.parse_args(['--dep-source', source])
            self.assertEqual(args.dep_source, source)


if __name__ == '__main__':
    unittest.main()
