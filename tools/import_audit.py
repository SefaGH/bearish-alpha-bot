#!/usr/bin/env python3
"""
Runtime Import Audit Tool

This module tracks which Python modules are actually imported during runtime.
It monitors sys.modules to see what gets loaded.
"""
import sys
import json
import os
import atexit
from datetime import datetime


class ImportAuditor:
    """Tracks runtime imports by monitoring sys.modules."""
    
    def __init__(self, output_file='artifacts/import_audit.json'):
        self.output_file = output_file
        self.start_time = datetime.now().isoformat()
        # Capture initial modules
        self.initial_modules = set(sys.modules.keys())
        
    def get_imported_modules(self):
        """Get all modules imported since initialization."""
        current_modules = set(sys.modules.keys())
        new_modules = current_modules - self.initial_modules
        return new_modules
    
    def save_audit(self):
        """Save the audit results to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            imported = self.get_imported_modules()
            
            # Categorize imports
            project_imports = []
            stdlib_imports = []
            external_imports = []
            
            for module_name in sorted(imported):
                # Skip private/internal modules
                if module_name.startswith('_'):
                    continue
                    
                # Check if it's a project module
                if any(module_name.startswith(prefix) for prefix in 
                       ['core.', 'strategies.', 'ml.', 'backtest.', 'scanners.', 'monitoring.', 
                        'universe', 'main', 'scripts.']):
                    project_imports.append(module_name)
                # Check if it's stdlib (simple heuristic)
                elif '.' not in module_name or module_name.split('.')[0] in [
                    'os', 'sys', 'asyncio', 'json', 'datetime', 'collections', 
                    'itertools', 'functools', 'pathlib', 'logging', 'time',
                    'unittest', 'contextlib', 're', 'io', 'pickle', 'hashlib',
                    'warnings', 'traceback', 'typing', 'enum', 'dataclasses'
                ]:
                    stdlib_imports.append(module_name)
                else:
                    external_imports.append(module_name)
            
            audit_data = {
                'metadata': {
                    'start_time': self.start_time,
                    'end_time': datetime.now().isoformat(),
                    'total_unique_imports': len(imported),
                    'project_imports': len(project_imports),
                    'stdlib_imports': len(stdlib_imports),
                    'external_imports': len(external_imports)
                },
                'project_modules': sorted(project_imports),
                'stdlib_modules': sorted(stdlib_imports)[:100],  # Limit stdlib
                'external_modules': sorted(external_imports),
                'all_imports': sorted(list(imported))[:500]  # Limit total
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
                
            print(f"[ImportAudit] Saved audit to {self.output_file}")
            print(f"[ImportAudit] Tracked {len(imported)} unique modules")
            print(f"[ImportAudit] - Project: {len(project_imports)}, Stdlib: {len(stdlib_imports)}, External: {len(external_imports)}")
            
        except Exception as e:
            print(f"[ImportAudit] Error saving audit: {e}")


# Global auditor instance
_auditor = None


def install(output_file='artifacts/import_audit.json'):
    """Install the import auditor."""
    global _auditor
    
    if _auditor is None:
        _auditor = ImportAuditor(output_file)
        
        # Register cleanup on exit
        atexit.register(_auditor.save_audit)
        
        print(f"[ImportAudit] Import auditor installed, will save to {output_file}")
    
    return _auditor


def get_auditor():
    """Get the current auditor instance."""
    return _auditor


# Auto-install when imported
if __name__ != "__main__":
    install()
