import json
import os

def analyze_appconfig_dump(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    settings = data.get('settings', [])
    
    report_lines = []
    report_lines.append("# Azure App Configuration Content-Type Hygiene Audit")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append(f"- **Total Keys Scanned**: {len(settings)}")
    
    complex_json_count = 0
    empty_ct_count = 0
    issues_count = 0
    
    detailed_rows = []

    issues = []
    
    for item in settings:
        key = item['key']
        value = item['value']
        content_type = item.get('contentType', '') or ''
        
        is_complex_json = False
        is_valid_json = False
        json_type = None

        # Check if it looks like a complex JSON structure (Object or Array)
        if value.strip().startswith('{') or value.strip().startswith('['):
            is_complex_json = True
            complex_json_count += 1
        
        # Try parsing
        try:
            parsed = json.loads(value)
            is_valid_json = True
            if isinstance(parsed, (dict, list)):
                json_type = 'complex'
            else:
                json_type = 'simple' # number, bool, string, null
        except (json.JSONDecodeError, TypeError):
            is_valid_json = False

        # Analysis Logic
        status = "OK"
        recommendation = "None"
        
        # Hygiene Check 1: Empty Content-Type
        if content_type == "":
            empty_ct_count += 1
            status = "MISSING_METADATA"
            recommendation = "Set to text/plain (or application/json if typed)"

        # Rule 1: Complex JSON (Dict/List) MUST be application/json
        if json_type == 'complex':
            if content_type != 'application/json':
                status = "CRITICAL_MISSING_HEADER"
                recommendation = "Set Content-Type to application/json"
            else:
                status = "OK (JSON)"
                recommendation = "Keep as is"
        
        # Rule 2: Marked as application/json MUST be valid JSON
        elif content_type == 'application/json':
            if not is_valid_json:
                status = "INVALID_JSON_CONTENT"
                recommendation = "Fix JSON syntax or remove Content-Type"
            elif json_type == 'simple':
                status = "SIMPLE_TYPE_AS_JSON"
                recommendation = "Review (Valid but usually text/plain is sufficient)"

        # Rule 3: Comma-separated lists (Modernization Opportunity)
        elif ',' in value and not is_complex_json and content_type in ['', 'text/plain']:
             # Check if it looks like a list of numbers or strings
             parts = value.split(',')
             if len(parts) > 1:
                 status = "POTENTIAL_JSON_ARRAY"
                 recommendation = "Consider converting to JSON Array [...]"

        if status != "OK" and status != "OK (JSON)":
            issues_count += 1
            issues.append({
                "key": key,
                "value": value,
                "current_ct": content_type,
                "status": status,
                "recommendation": recommendation
            })

        # Escape pipes for markdown table
        safe_value = (value[:40] + '...') if len(value) > 40 else value
        safe_value = safe_value.replace('|', '\|').replace('\n', ' ')
        
        # Add to detailed rows
        detailed_rows.append(f"| `{key}` | `{safe_value}` | `{content_type}` | {status} | {recommendation} |")

    report_lines.append(f"- **Complex JSON Objects**: {complex_json_count}")
    report_lines.append(f"- **Empty Content-Type**: {empty_ct_count}")
    report_lines.append(f"- **Issues/Suggestions**: {issues_count}")
    report_lines.append("")
    
    report_lines.append("## Detailed Inventory")
    report_lines.append("| Key | Value Preview | Content-Type | Status | Recommendation |")
    report_lines.append("|---|---|---|---|---|")
    report_lines.extend(detailed_rows)

    # Write Report
    with open('APPCONFIG_CONTENTTYPE_JSON_AUDIT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Analysis complete. Found {issues_count} issues/suggestions.")
    
    # Generate Fix Script for Empty Content Types (Safe default: text/plain)
    # We will NOT auto-fix "POTENTIAL_JSON_ARRAY" as that requires code changes.
    fix_script_lines = []
    fix_script_lines.append("#!/bin/bash")
    fix_script_lines.append("# Auto-generated fix script for Content-Type Hygiene")
    
    for issue in issues:
        if issue['status'] == 'MISSING_METADATA':
            # Default to text/plain for safety
            cmd = f'az appconfig kv set --name appcs-bearish-bot --key "{issue["key"]}" --content-type "text/plain" --yes'
            fix_script_lines.append(cmd)
            
    with open('fix_content_types.sh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fix_script_lines))

if __name__ == "__main__":
    analyze_appconfig_dump('appconfig_dump.json')
