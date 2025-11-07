"""
Pre-Training Data Diagnostics Script

This script analyzes training data quality before model training begins.
It performs comprehensive checks on data fetched from BingX and generates
detailed reports with warnings and recommendations.

Quality Checks:
- Data completeness (percentage of expected candles present)
- Missing values detection
- Duplicate detection
- Time consistency (gaps in timestamps)
- Overall quality score (0-100)

Outputs:
- Markdown report: logs/diagnostics/latest_diagnostics.md
- JSON report: logs/diagnostics/pre_training_diagnostics_{timestamp}.json

Usage:
    python scripts/diagnose_training_data.py
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.ccxt_client import CcxtClient
from src.core.logger import setup_logger

# Setup logger
logger = setup_logger("data-diagnostics", level=logging.INFO, log_to_file=True)

# Configuration
SYMBOLS_TO_CHECK = ['BTC/USDT']
TIMEFRAMES_TO_CHECK = ['5m', '15m', '30m', '1h', '4h']
EXPECTED_CANDLES = 1440  # BingX API limit
DIAGNOSTICS_DIR = Path("logs/diagnostics")


class DataQualityDiagnostics:
    """
    Performs comprehensive data quality checks on training data.
    """
    
    def __init__(self):
        """Initialize diagnostics."""
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "symbols_checked": [],
            "timeframes_checked": [],
            "overall_quality_score": 0.0,
            "warnings": [],
            "recommendations": [],
            "detailed_results": {}
        }
        logger.info("DataQualityDiagnostics initialized")
    
    def check_completeness(self, df: pd.DataFrame, expected_count: int) -> Tuple[float, Dict[str, Any]]:
        """
        Check data completeness.
        
        Args:
            df: DataFrame to check
            expected_count: Expected number of candles
        
        Returns:
            Tuple of (completeness_pct, details_dict)
        """
        actual_count = len(df)
        completeness_pct = (actual_count / expected_count) * 100 if expected_count > 0 else 0.0
        
        details = {
            "expected_candles": expected_count,
            "actual_candles": actual_count,
            "missing_candles": expected_count - actual_count,
            "completeness_pct": completeness_pct
        }
        
        return completeness_pct, details
    
    def check_missing_values(self, df: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
        """
        Check for missing values in the DataFrame.
        
        Args:
            df: DataFrame to check
        
        Returns:
            Tuple of (total_missing, details_dict)
        """
        missing_by_column = df.isnull().sum().to_dict()
        total_missing = sum(missing_by_column.values())
        
        details = {
            "total_missing_values": total_missing,
            "missing_by_column": {k: int(v) for k, v in missing_by_column.items() if v > 0}
        }
        
        return total_missing, details
    
    def check_duplicates(self, df: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
        """
        Check for duplicate rows based on timestamp.
        
        Args:
            df: DataFrame to check
        
        Returns:
            Tuple of (duplicate_count, details_dict)
        """
        if 'timestamp' in df.columns:
            duplicate_count = df.duplicated(subset=['timestamp'], keep='first').sum()
        else:
            # Check index if timestamp column not present
            duplicate_count = df.index.duplicated(keep='first').sum()
        
        details = {
            "duplicate_rows": int(duplicate_count),
            "has_duplicates": duplicate_count > 0
        }
        
        return duplicate_count, details
    
    def check_time_consistency(self, df: pd.DataFrame, timeframe: str) -> Tuple[int, Dict[str, Any]]:
        """
        Check for gaps in time series data.
        
        Args:
            df: DataFrame with datetime index
            timeframe: Timeframe string (e.g., '5m', '1h')
        
        Returns:
            Tuple of (gap_count, details_dict)
        """
        # Parse timeframe to timedelta
        timeframe_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        expected_freq = timeframe_map.get(timeframe, '1H')
        
        if len(df) < 2:
            return 0, {"gap_count": 0, "gaps": []}
        
        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(expected_freq)
        
        # Find gaps (differences larger than expected)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Allow 50% tolerance
        gap_count = len(gaps)
        
        details = {
            "gap_count": gap_count,
            "expected_frequency": expected_freq,
            "gaps": [str(idx) for idx in gaps.index[:5]]  # First 5 gaps
        }
        
        return gap_count, details
    
    def calculate_quality_score(self, 
                                completeness_pct: float,
                                missing_count: int,
                                duplicate_count: int,
                                gap_count: int,
                                total_rows: int) -> float:
        """
        Calculate overall quality score (0-100).
        
        Scoring:
        - Completeness: 40 points (linear scale)
        - Missing values: 30 points (deduct based on percentage)
        - Duplicates: 20 points (deduct based on percentage)
        - Time gaps: 10 points (deduct based on count)
        
        Args:
            completeness_pct: Completeness percentage
            missing_count: Number of missing values
            duplicate_count: Number of duplicate rows
            gap_count: Number of time gaps
            total_rows: Total number of rows
        
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Completeness score (0-40 points)
        score += (completeness_pct / 100) * 40
        
        # Missing values penalty (0-30 points)
        if total_rows > 0:
            missing_pct = (missing_count / (total_rows * 6)) * 100  # Assuming 6 columns (OHLCV + timestamp)
            missing_score = max(0, 30 - (missing_pct * 3))  # Deduct 3 points per 1% missing
            score += missing_score
        else:
            score += 30
        
        # Duplicates penalty (0-20 points)
        if total_rows > 0:
            duplicate_pct = (duplicate_count / total_rows) * 100
            duplicate_score = max(0, 20 - (duplicate_pct * 10))  # Deduct 10 points per 1% duplicates
            score += duplicate_score
        else:
            score += 20
        
        # Time gaps penalty (0-10 points)
        gap_score = max(0, 10 - (gap_count * 0.5))  # Deduct 0.5 points per gap
        score += gap_score
        
        return min(100, max(0, score))
    
    async def diagnose_symbol_timeframe(self, 
                                       exchange_client: CcxtClient,
                                       symbol: str,
                                       timeframe: str) -> Dict[str, Any]:
        """
        Perform comprehensive diagnostics on a symbol-timeframe pair.
        
        Args:
            exchange_client: CCXT exchange client
            symbol: Trading symbol
            timeframe: Timeframe string
        
        Returns:
            Diagnostic results dictionary
        """
        logger.info(f"📊 Diagnosing: {symbol} [{timeframe}]")
        
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "success",
            "error": None,
            "checks": {}
        }
        
        try:
            # Fetch data
            df = await exchange_client.ohlcv(
                symbol, 
                timeframe=timeframe, 
                limit=EXPECTED_CANDLES, 
                add_indicators=False
            )
            
            if df is None or df.empty:
                result["status"] = "failed"
                result["error"] = "No data returned"
                logger.warning(f"❌ No data for {symbol} {timeframe}")
                return result
            
            # Run checks
            completeness_pct, completeness_details = self.check_completeness(df, EXPECTED_CANDLES)
            missing_count, missing_details = self.check_missing_values(df)
            duplicate_count, duplicate_details = self.check_duplicates(df)
            gap_count, gap_details = self.check_time_consistency(df, timeframe)
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(
                completeness_pct, missing_count, duplicate_count, gap_count, len(df)
            )
            
            # Store results
            result["checks"] = {
                "completeness": completeness_details,
                "missing_values": missing_details,
                "duplicates": duplicate_details,
                "time_consistency": gap_details,
                "quality_score": quality_score
            }
            
            # Generate warnings
            warnings = []
            if completeness_pct < 90:
                warnings.append(f"Low completeness: {completeness_pct:.1f}% (expected ≥90%)")
            if quality_score < 80:
                warnings.append(f"Low quality score: {quality_score:.1f} (expected ≥80)")
            if missing_count > 0:
                warnings.append(f"Missing values detected: {missing_count}")
            if duplicate_count > 0:
                warnings.append(f"Duplicate rows detected: {duplicate_count}")
            if gap_count > 10:
                warnings.append(f"Multiple time gaps: {gap_count}")
            
            result["warnings"] = warnings
            
            # Log summary
            status_emoji = "✅" if quality_score >= 80 else "⚠️" if quality_score >= 60 else "❌"
            logger.info(
                f"{status_emoji} {symbol} {timeframe}: "
                f"Quality={quality_score:.1f}, "
                f"Completeness={completeness_pct:.1f}%, "
                f"Missing={missing_count}, "
                f"Duplicates={duplicate_count}, "
                f"Gaps={gap_count}"
            )
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"❌ Error diagnosing {symbol} {timeframe}: {e}", exc_info=True)
        
        return result
    
    async def run_diagnostics(self):
        """
        Run diagnostics on all configured symbols and timeframes.
        """
        logger.info("="*60)
        logger.info("🔍 STARTING PRE-TRAINING DATA DIAGNOSTICS")
        logger.info("="*60)
        
        # Initialize exchange client
        exchange_client = CcxtClient('bingx')
        
        # Store configuration
        self.results["symbols_checked"] = SYMBOLS_TO_CHECK
        self.results["timeframes_checked"] = TIMEFRAMES_TO_CHECK
        
        # Run diagnostics for each symbol-timeframe pair
        total_checks = 0
        total_quality = 0.0
        
        for symbol in SYMBOLS_TO_CHECK:
            for timeframe in TIMEFRAMES_TO_CHECK:
                result = await self.diagnose_symbol_timeframe(
                    exchange_client, symbol, timeframe
                )
                
                # Store result
                key = f"{symbol}_{timeframe}"
                self.results["detailed_results"][key] = result
                
                # Accumulate for overall score
                if result["status"] == "success" and "quality_score" in result["checks"]:
                    total_quality += result["checks"]["quality_score"]
                    total_checks += 1
                
                # Add warnings to overall list
                if result.get("warnings"):
                    for warning in result["warnings"]:
                        self.results["warnings"].append(f"{symbol} {timeframe}: {warning}")
        
        # Calculate overall quality score
        if total_checks > 0:
            self.results["overall_quality_score"] = total_quality / total_checks
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"✅ Diagnostics complete. Overall quality: {self.results['overall_quality_score']:.1f}/100")
    
    def _generate_recommendations(self):
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        overall_quality = self.results["overall_quality_score"]
        
        if overall_quality < 70:
            recommendations.append(
                "⚠️ CRITICAL: Overall data quality is below acceptable threshold. "
                "Consider investigating data source issues before training."
            )
        elif overall_quality < 80:
            recommendations.append(
                "⚠️ WARNING: Data quality is moderate. Training may proceed but results could be suboptimal."
            )
        else:
            recommendations.append(
                "✅ Data quality is good. Safe to proceed with training."
            )
        
        if len(self.results["warnings"]) > 10:
            recommendations.append(
                f"Multiple quality issues detected ({len(self.results['warnings'])} warnings). "
                "Review detailed diagnostics."
            )
        
        # Check for specific issues
        has_completeness_issues = any("completeness" in w.lower() for w in self.results["warnings"])
        has_missing_values = any("missing" in w.lower() for w in self.results["warnings"])
        has_time_gaps = any("gap" in w.lower() for w in self.results["warnings"])
        
        if has_completeness_issues:
            recommendations.append(
                "Some datasets have low completeness. Consider increasing data fetch limits or checking API connection."
            )
        
        if has_missing_values:
            recommendations.append(
                "Missing values detected. Ensure feature engineering pipeline handles missing data properly."
            )
        
        if has_time_gaps:
            recommendations.append(
                "Time gaps detected in data. This may affect time-series models. Consider data interpolation."
            )
        
        self.results["recommendations"] = recommendations
    
    def save_json_report(self) -> str:
        """
        Save diagnostics report as JSON.
        
        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = DIAGNOSTICS_DIR / f"pre_training_diagnostics_{timestamp}.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved JSON report: {json_file}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
        
        return str(json_file)
    
    def save_markdown_report(self) -> str:
        """
        Save diagnostics report as Markdown.
        
        Returns:
            Path to saved Markdown file
        """
        md_file = DIAGNOSTICS_DIR / "latest_diagnostics.md"
        
        lines = [
            "# Pre-Training Data Diagnostics Report",
            "",
            f"**Generated:** {self.results['timestamp']}",
            f"**Overall Quality Score:** {self.results['overall_quality_score']:.1f}/100",
            "",
            "---",
            ""
        ]
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for rec in self.results["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")
        
        # Warnings Summary
        if self.results["warnings"]:
            lines.append("## Warnings")
            lines.append("")
            for warning in self.results["warnings"]:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")
        
        # Detailed Results
        lines.append("## Detailed Results")
        lines.append("")
        
        for key, result in self.results["detailed_results"].items():
            symbol, timeframe = key.rsplit('_', 1)
            lines.append(f"### {symbol} - {timeframe}")
            lines.append("")
            
            if result["status"] != "success":
                lines.append(f"**Status:** ❌ {result['status']}")
                lines.append(f"**Error:** {result.get('error', 'Unknown')}")
            else:
                checks = result["checks"]
                quality_score = checks.get("quality_score", 0)
                
                emoji = "✅" if quality_score >= 80 else "⚠️" if quality_score >= 60 else "❌"
                lines.append(f"**Quality Score:** {emoji} {quality_score:.1f}/100")
                lines.append("")
                
                # Completeness
                comp = checks.get("completeness", {})
                lines.append(f"- **Completeness:** {comp.get('completeness_pct', 0):.1f}% "
                           f"({comp.get('actual_candles', 0)}/{comp.get('expected_candles', 0)} candles)")
                
                # Missing values
                missing = checks.get("missing_values", {})
                lines.append(f"- **Missing Values:** {missing.get('total_missing_values', 0)}")
                
                # Duplicates
                dup = checks.get("duplicates", {})
                lines.append(f"- **Duplicates:** {dup.get('duplicate_rows', 0)}")
                
                # Time gaps
                gaps = checks.get("time_consistency", {})
                lines.append(f"- **Time Gaps:** {gaps.get('gap_count', 0)}")
                
                # Warnings
                if result.get("warnings"):
                    lines.append("")
                    lines.append("**Issues:**")
                    for warning in result["warnings"]:
                        lines.append(f"  - {warning}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Save to file
        report = "\n".join(lines)
        
        try:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✅ Saved Markdown report: {md_file}")
        except Exception as e:
            logger.error(f"Failed to save Markdown report: {e}")
        
        return str(md_file)


async def main():
    """Main execution function."""
    diagnostics = DataQualityDiagnostics()
    
    try:
        # Run diagnostics
        await diagnostics.run_diagnostics()
        
        # Save reports
        json_path = diagnostics.save_json_report()
        md_path = diagnostics.save_markdown_report()
        
        logger.info("="*60)
        logger.info("✅ PRE-TRAINING DIAGNOSTICS COMPLETE")
        logger.info(f"   JSON Report: {json_path}")
        logger.info(f"   Markdown Report: {md_path}")
        logger.info(f"   Overall Quality: {diagnostics.results['overall_quality_score']:.1f}/100")
        logger.info("="*60)
        
        # Exit with appropriate code
        if diagnostics.results['overall_quality_score'] < 70:
            logger.warning("⚠️ Data quality below threshold. Review diagnostics before training.")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Diagnostics failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
