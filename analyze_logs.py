import json
import glob
import os
from datetime import datetime

def parse_timestamp(ts_str):
    try:
        # Handle Z at the end
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1]
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None

def analyze_logs(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    print(f"Found {len(log_files)} log files.")

    runs = {}

    for file_path in log_files:
        print(f"Processing {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if 'properties' not in entry:
                    continue

                props = entry['properties']
                resource = props.get('resource', {})
                run_id = resource.get('runId')
                action_name = resource.get('actionName', 'Unknown')
                
                if not run_id:
                    continue

                if run_id not in runs:
                    runs[run_id] = {
                        'start_time': None,
                        'end_time': None,
                        'actions': {},
                        'status': 'Unknown',
                        'errors': []
                    }

                # Update run timing
                timestamp = parse_timestamp(entry.get('time'))
                if timestamp:
                    if runs[run_id]['start_time'] is None or timestamp < runs[run_id]['start_time']:
                        runs[run_id]['start_time'] = timestamp
                    if runs[run_id]['end_time'] is None or timestamp > runs[run_id]['end_time']:
                        runs[run_id]['end_time'] = timestamp

                # Track action status
                op_name = entry.get('operationName')
                
                if op_name == 'Microsoft.Logic/workflows/workflowActionCompleted':
                    status = props.get('status')
                    code = props.get('code')
                    
                    runs[run_id]['actions'][action_name] = status
                    
                    if status == 'Failed':
                        error_msg = props.get('error', {}).get('message', 'No error message')
                        runs[run_id]['errors'].append(f"Action '{action_name}' failed: {error_msg}")
                        runs[run_id]['status'] = 'Failed'

    # Generate Report
    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("="*50)
    
    sorted_runs = sorted(runs.items(), key=lambda x: x[1]['start_time'] if x[1]['start_time'] else datetime.min)

    for run_id, data in sorted_runs:
        print(f"\nRun ID: {run_id}")
        print(f"Time: {data['start_time']} - {data['end_time']}")
        
        if data['errors']:
            print("Status: ❌ FAILED")
            for err in data['errors']:
                print(f"  - {err}")
        else:
            print("Status: ✅ SUCCESS (or In Progress)")
        
        # Check for specific critical actions
        critical_actions = ['Start_Automation_Runbook', 'Get_Job_Status', 'HTTP']
        print("Critical Actions:")
        for action in critical_actions:
            status = data['actions'].get(action, 'Not Started')
            icon = "✅" if status == 'Succeeded' else "❌" if status == 'Failed' else "⚠️" if status == 'Skipped' else "⏳"
            print(f"  - {action}: {icon} {status}")

if __name__ == "__main__":
    analyze_logs("downloaded_logs")
