#!/usr/bin/env python3
"""
Analyze unused file candidates by combining:
1. Coverage data (low/zero coverage files)
2. Runtime import audit (files never imported)
3. Static import analysis (files not referenced)
"""
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_coverage_xml(coverage_file):
    """Parse coverage.xml and extract files with low coverage."""
    low_coverage_files = []
    
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        # Find all classes (files) with their coverage
        for package in root.findall('.//package'):
            for cls in package.findall('classes/class'):
                filename = cls.get('filename')
                line_rate = float(cls.get('line-rate', 0))
                
                # Files with less than 10% coverage or 0%
                if line_rate < 0.1:
                    low_coverage_files.append({
                        'file': filename,
                        'coverage': line_rate * 100,
                        'reason': f'coverage={line_rate*100:.1f}%'
                    })
    except Exception as e:
        print(f"Warning: Could not parse coverage XML: {e}")
    
    return low_coverage_files


def load_runtime_imports(import_audit_file):
    """Load runtime import audit data."""
    try:
        with open(import_audit_file, 'r') as f:
            data = json.load(f)
            return set(data.get('project_modules', []))
    except Exception as e:
        print(f"Warning: Could not load import audit: {e}")
        return set()


def find_python_files(base_dir):
    """Find all Python files in src, scripts, tests directories."""
    python_files = []
    
    for directory in ['src', 'scripts', 'tests']:
        dir_path = Path(base_dir) / directory
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                # Skip __pycache__ and similar
                if '__pycache__' not in str(py_file):
                    python_files.append(py_file)
    
    return python_files


def module_name_from_path(file_path, base_dir):
    """Convert file path to Python module name."""
    try:
        rel_path = file_path.relative_to(Path(base_dir))
        
        # Remove .py extension
        module_path = str(rel_path)[:-3] if str(rel_path).endswith('.py') else str(rel_path)
        
        # Convert path to module notation
        # Handle different base directories
        parts = module_path.split(os.sep)
        
        # Remove directory prefix (src, scripts, tests)
        if parts[0] in ['src', 'scripts', 'tests']:
            parts = parts[1:]
        
        # Replace __init__ with package name
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        module_name = '.'.join(parts)
        return module_name
    except Exception:
        return None


def analyze_unused_candidates(base_dir):
    """Combine all analysis sources to find unused file candidates."""
    
    # Load coverage data
    coverage_file = os.path.join(base_dir, 'artifacts', 'coverage.xml')
    low_coverage = parse_coverage_xml(coverage_file)
    
    # Load runtime imports
    import_audit_file = os.path.join(base_dir, 'artifacts', 'import_audit.json')
    runtime_imports = load_runtime_imports(import_audit_file)
    
    # Find all Python files
    all_python_files = find_python_files(base_dir)
    
    # Analyze each file
    candidates = []
    
    for py_file in all_python_files:
        reasons = []
        module_name = module_name_from_path(py_file, base_dir)
        rel_path = str(py_file.relative_to(base_dir))
        
        # Check if in runtime imports
        if module_name:
            # Check exact match or parent match
            imported = any(
                runtime_import.startswith(module_name) or module_name.startswith(runtime_import)
                for runtime_import in runtime_imports
            )
            
            if not imported:
                reasons.append('not_imported_runtime')
        
        # Check coverage (if available)
        coverage_info = next((c for c in low_coverage if c['file'] in str(py_file)), None)
        if coverage_info:
            if coverage_info['coverage'] == 0:
                reasons.append('coverage=0%')
            else:
                reasons.append(f"coverage={coverage_info['coverage']:.1f}%")
        
        # Special cases to exclude from candidates
        exclude = False
        
        # Don't flag test files (they're not supposed to be imported)
        if rel_path.startswith('tests/'):
            exclude = True
        
        # Don't flag __init__.py files
        if py_file.name == '__init__.py':
            exclude = True
            
        # Don't flag main entry points
        if py_file.name in ['main.py', 'universe.py', '__main__.py']:
            exclude = True
        
        # Don't flag scripts (they're meant to be run, not imported)
        if rel_path.startswith('scripts/') and py_file.name not in ['__init__.py']:
            # Scripts are executables, not libraries
            exclude = True
        
        # Add to candidates if there are reasons and not excluded
        if reasons and not exclude:
            candidates.append({
                'path': rel_path,
                'module': module_name or 'N/A',
                'reasons': reasons
            })
    
    return sorted(candidates, key=lambda x: (len(x['reasons']), x['path']), reverse=True)


def main():
    """Main entry point."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Analyzing unused file candidates...")
    print(f"Base directory: {base_dir}")
    print()
    
    candidates = analyze_unused_candidates(base_dir)
    
    # Write to output file
    output_file = os.path.join(base_dir, 'artifacts', 'unused_candidates.txt')
    
    with open(output_file, 'w') as f:
        f.write("# Unused File Candidates\n")
        f.write("# Format: relative/path — reason tags\n")
        f.write("# This is an automated analysis - manual review required before removal\n")
        f.write(f"# Total candidates found: {len(candidates)}\n")
        f.write("\n")
        
        if candidates:
            for candidate in candidates:
                reasons_str = '; '.join(candidate['reasons'])
                f.write(f"{candidate['path']} — {reasons_str}\n")
        else:
            f.write("# No unused candidates found (good news!)\n")
    
    print(f"Found {len(candidates)} unused file candidates")
    print(f"Results written to: {output_file}")
    print()
    
    # Show top 10
    if candidates:
        print("Top candidates:")
        for i, candidate in enumerate(candidates[:10], 1):
            reasons_str = '; '.join(candidate['reasons'])
            print(f"  {i}. {candidate['path']}")
            print(f"     Reasons: {reasons_str}")
    
    return 0


if __name__ == '__main__':
    exit(main())
