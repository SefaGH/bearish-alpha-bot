#!/usr/bin/env python3
"""
Test File Aging Analysis Script

This script analyzes the age of test files in the tests/ directory by examining
their last modification dates using git log. It helps identify technical debt
candidates by showing which test files haven't been updated in a long time.

Usage:
    python scripts/analyze_test_age.py [--output OUTPUT_FILE] [--days DAYS]

Options:
    --output OUTPUT_FILE    Write report to file (default: stdout)
    --days DAYS            Show only files older than DAYS (default: all)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
import argparse
import re


def get_git_file_info(file_path: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Get the last commit information for a file.
    
    Args:
        file_path: Path to the file relative to repo root
        
    Returns:
        Tuple of (date, author, commit_hash, subject) or None if error
    """
    try:
        # Get last commit info for the file
        cmd = [
            'git', 'log', '-1',
            '--format=%ci,%an,%h,%s',
            '--', file_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if not result.stdout.strip():
            # File might be new and not yet committed
            return None
            
        # Parse the output: date,author,hash,subject
        parts = result.stdout.strip().split(',', 3)
        if len(parts) >= 4:
            return (parts[0], parts[1], parts[2], parts[3])
        return None
        
    except subprocess.CalledProcessError:
        return None


def analyze_test_files(repo_root: Path, min_days: Optional[int] = None) -> List[dict]:
    """
    Analyze all test files and get their aging information.
    
    Args:
        repo_root: Path to repository root
        min_days: Only include files older than this many days (optional)
        
    Returns:
        List of dicts with file info sorted by age (oldest first)
    """
    tests_dir = repo_root / 'tests'
    
    if not tests_dir.exists():
        print(f"Error: tests/ directory not found at {tests_dir}", file=sys.stderr)
        return []
    
    # Find all Python test files
    test_files = sorted(tests_dir.rglob('*.py'))
    
    file_info = []
    
    for test_file in test_files:
        rel_path = test_file.relative_to(repo_root)
        info = get_git_file_info(str(rel_path))
        
        if info is None:
            # File not in git history yet
            continue
            
        date_str, author, commit_hash, subject = info
        
        # Parse date with timezone (format: 2024-11-18 16:58:22 +0000 or 2024-11-18 16:58:22 -0500)
        try:
            # Extract timezone offset
            tz_match = re.search(r'([+-]\d{4})$', date_str)
            if tz_match:
                # Parse the datetime with timezone
                from datetime import timezone, timedelta
                tz_str = tz_match.group(1)
                hours_offset = int(tz_str[1:3])
                minutes_offset = int(tz_str[3:5])
                if tz_str[0] == '-':
                    hours_offset = -hours_offset
                    minutes_offset = -minutes_offset
                tz = timezone(timedelta(hours=hours_offset, minutes=minutes_offset))
                
                # Remove timezone from string and parse
                date_part = date_str[:date_str.rfind(' ')]
                commit_date = datetime.strptime(date_part, '%Y-%m-%d %H:%M:%S')
                commit_date = commit_date.replace(tzinfo=tz)
                
                # Convert to UTC for comparison
                commit_date_utc = commit_date.astimezone(timezone.utc).replace(tzinfo=None)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                days_old = (now_utc - commit_date_utc).days
            else:
                # Fallback: no timezone
                commit_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                days_old = (datetime.now() - commit_date).days
            
            # Filter by minimum days if specified
            if min_days is not None and days_old < min_days:
                continue
            
            file_info.append({
                'file': str(rel_path),
                'date': commit_date,
                'days_old': days_old,
                'author': author,
                'commit': commit_hash,
                'subject': subject
            })
        except ValueError:
            # Skip if date parsing fails
            continue
    
    # Sort by age (oldest first)
    file_info.sort(key=lambda x: x['date'])
    
    return file_info


def generate_report(file_info: List[dict], output_file: Optional[str] = None):
    """
    Generate and output the aging report.
    
    Args:
        file_info: List of file information dicts
        output_file: Optional file path to write report to
    """
    if output_file:
        f = open(output_file, 'w')
    else:
        f = sys.stdout
    
    try:
        # Header
        f.write("=" * 100 + "\n")
        f.write("TEST FILE AGING ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files analyzed: {len(file_info)}\n")
        f.write("=" * 100 + "\n\n")
        
        if not file_info:
            f.write("No test files found or all files are too recent.\n")
            return
        
        # Summary statistics
        ages = [info['days_old'] for info in file_info]
        avg_age = sum(ages) / len(ages)
        oldest = max(ages)
        newest = min(ages)
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"  Oldest file: {oldest} days\n")
        f.write(f"  Newest file: {newest} days\n")
        f.write(f"  Average age: {avg_age:.1f} days\n")
        f.write(f"  Files > 180 days old: {sum(1 for age in ages if age > 180)}\n")
        f.write(f"  Files > 90 days old: {sum(1 for age in ages if age > 90)}\n")
        f.write(f"  Files > 30 days old: {sum(1 for age in ages if age > 30)}\n")
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Detailed file list
        f.write("DETAILED FILE LIST (Oldest to Newest):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'File':<50} {'Days':<8} {'Last Modified':<20} {'Author':<20}\n")
        f.write("-" * 100 + "\n")
        
        for info in file_info:
            file_short = info['file']
            if len(file_short) > 47:
                file_short = "..." + file_short[-44:]
            
            author_short = info['author']
            if len(author_short) > 17:
                author_short = author_short[:14] + "..."
                
            f.write(
                f"{file_short:<50} {info['days_old']:<8} "
                f"{info['date'].strftime('%Y-%m-%d %H:%M'):<20} {author_short:<20}\n"
            )
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Top 20 oldest files with commit info
        f.write("TOP 20 OLDEST FILES (Potential Technical Debt Candidates):\n")
        f.write("-" * 100 + "\n")
        
        for i, info in enumerate(file_info[:20], 1):
            f.write(f"\n{i}. {info['file']}\n")
            f.write(f"   Age: {info['days_old']} days ({info['date'].strftime('%Y-%m-%d')})\n")
            f.write(f"   Author: {info['author']}\n")
            f.write(f"   Commit: {info['commit']}\n")
            f.write(f"   Message: {info['subject']}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        
    finally:
        if output_file:
            f.close()
            print(f"Report written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze test file ages to identify technical debt candidates'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        help='Show only files older than this many days'
    )
    
    args = parser.parse_args()
    
    # Get repo root (assume script is in scripts/ subdirectory)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    print(f"Analyzing test files in: {repo_root / 'tests'}")
    print("This may take a moment...\n")
    
    # Analyze files
    file_info = analyze_test_files(repo_root, args.days)
    
    # Generate report
    generate_report(file_info, args.output)


if __name__ == '__main__':
    main()
