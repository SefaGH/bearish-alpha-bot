"""
Model Performance Tracker for ML Training

This module tracks and records model training metrics, compares with previous
training runs, and generates detailed reports in multiple formats (JSON, CSV, Markdown).

Key Features:
- Records training metrics (accuracy, precision, recall, f1_score, loss)
- Maintains historical training records
- Compares current training with previous runs
- Generates summary reports in Markdown format
- Exports to CSV for analysis
- Git SHA and run number versioning

Usage:
    tracker = ModelPerformanceTracker()
    tracker.record_training(
        model_type="regime",
        model_name="BTC-USDT_30m",
        metrics={...},
        data_info={...},
        training_time=120.5
    )
    tracker.generate_summary_report()
    tracker.export_to_csv()
"""

import json
import csv
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """
    Tracks and records ML model training performance metrics.
    
    Attributes:
        history_file: Path to JSON file storing training history
        csv_file: Path to CSV export file
        performance_dir: Directory for performance logs
    """
    
    def __init__(self, 
                 performance_dir: str = "logs/performance",
                 history_filename: str = "performance_history.json"):
        """
        Initialize the performance tracker.
        
        Args:
            performance_dir: Directory to store performance logs
            history_filename: Filename for the history JSON file
        """
        self.performance_dir = Path(performance_dir)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.performance_dir / history_filename
        self.csv_file = self.performance_dir / "performance_history.csv"
        
        # Load existing history or create new
        self.history = self._load_history()
        
        logger.info(f"ModelPerformanceTracker initialized: {self.history_file}")
    
    def _load_history(self) -> Dict[str, Any]:
        """
        Load existing training history from JSON file.
        
        Returns:
            Dictionary containing training history or empty structure
        """
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"Loaded {len(history.get('trainings', []))} previous training records")
                return history
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
                return {"trainings": []}
        else:
            logger.info("No previous history found, starting fresh")
            return {"trainings": []}
    
    def _save_history(self):
        """Save training history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved training history: {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def record_training(self,
                       model_type: str,
                       model_name: str,
                       metrics: Dict[str, float],
                       data_info: Dict[str, Any],
                       training_time: float,
                       git_sha: Optional[str] = None,
                       run_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Record a training session with all metrics and metadata.
        
        Args:
            model_type: Type of model (e.g., "regime", "rl", "signal", "price")
            model_name: Specific model name (e.g., "BTC-USDT_30m")
            metrics: Dictionary of training metrics
                     (accuracy, precision, recall, f1_score, loss, val_loss, etc.)
            data_info: Information about training data
                      (total_samples, train_samples, test_samples, features, etc.)
            training_time: Training duration in seconds
            git_sha: Git commit SHA (optional, will try to get from env)
            run_number: GitHub Actions run number (optional, will try to get from env)
        
        Returns:
            Dictionary containing the recorded training entry
        """
        # Get git SHA and run number from environment if not provided
        if git_sha is None:
            git_sha = os.environ.get('GITHUB_SHA', 'unknown')
        if run_number is None:
            run_number = os.environ.get('GITHUB_RUN_NUMBER', 'local')
        
        # Create training record
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "model_name": model_name,
            "git_sha": git_sha,
            "run_number": run_number,
            "metrics": metrics,
            "data_info": data_info,
            "training_time_seconds": training_time
        }
        
        # Add to history
        self.history["trainings"].append(training_record)
        
        # Save updated history
        self._save_history()
        
        # Also save individual model training record
        self._save_individual_record(model_type, model_name, training_record)
        
        logger.info(f"✅ Recorded training: {model_type}/{model_name} "
                   f"(accuracy={metrics.get('accuracy', 'N/A'):.4f}, "
                   f"time={training_time:.2f}s)")
        
        return training_record
    
    def _save_individual_record(self, model_type: str, model_name: str, 
                                record: Dict[str, Any]):
        """
        Save individual training record to separate file.
        
        Args:
            model_type: Type of model
            model_name: Specific model name
            record: Training record dictionary
        """
        # Create model-specific directory
        model_dir = self.performance_dir / model_type / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = model_dir / f"training_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved individual record: {filename}")
        except Exception as e:
            logger.error(f"Failed to save individual record: {e}")
    
    def compare_with_previous(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """
        Compare latest training with previous runs for the same model.
        
        Args:
            model_type: Type of model
            model_name: Specific model name
        
        Returns:
            Dictionary containing comparison results
        """
        # Get all trainings for this model
        model_trainings = [
            t for t in self.history["trainings"]
            if t["model_type"] == model_type and t["model_name"] == model_name
        ]
        
        if len(model_trainings) < 2:
            return {
                "comparison_available": False,
                "message": "Not enough training history for comparison"
            }
        
        # Get latest and previous
        latest = model_trainings[-1]
        previous = model_trainings[-2]
        
        # Compare metrics
        comparison = {
            "comparison_available": True,
            "latest_timestamp": latest["timestamp"],
            "previous_timestamp": previous["timestamp"],
            "metric_changes": {}
        }
        
        for metric_name in latest["metrics"]:
            if metric_name in previous["metrics"]:
                latest_val = latest["metrics"][metric_name]
                previous_val = previous["metrics"][metric_name]
                
                # Calculate change
                if previous_val != 0:
                    change_pct = ((latest_val - previous_val) / abs(previous_val)) * 100
                else:
                    change_pct = 0.0
                
                comparison["metric_changes"][metric_name] = {
                    "latest": latest_val,
                    "previous": previous_val,
                    "change": latest_val - previous_val,
                    "change_pct": change_pct,
                    "improved": latest_val > previous_val if "loss" not in metric_name.lower() 
                               else latest_val < previous_val
                }
        
        return comparison
    
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive summary report in Markdown format.
        
        Args:
            output_file: Optional path to save the report (default: logs/performance/training_report.md)
        
        Returns:
            Markdown formatted report string
        """
        if output_file is None:
            output_file = self.performance_dir / "training_report.md"
        else:
            output_file = Path(output_file)
        
        # Start building report
        lines = [
            "# ML Model Training Performance Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Training Sessions:** {len(self.history['trainings'])}",
            "",
            "---",
            ""
        ]
        
        if not self.history["trainings"]:
            lines.append("*No training sessions recorded yet.*")
        else:
            # Group by model type
            by_type = {}
            for training in self.history["trainings"]:
                model_type = training["model_type"]
                if model_type not in by_type:
                    by_type[model_type] = []
                by_type[model_type].append(training)
            
            # Report for each model type
            for model_type, trainings in by_type.items():
                lines.append(f"## {model_type.upper()} Models")
                lines.append("")
                
                # Group by model name
                by_name = {}
                for t in trainings:
                    name = t["model_name"]
                    if name not in by_name:
                        by_name[name] = []
                    by_name[name].append(t)
                
                for model_name, model_trainings in by_name.items():
                    latest = model_trainings[-1]
                    lines.append(f"### {model_name}")
                    lines.append("")
                    lines.append(f"- **Training Sessions:** {len(model_trainings)}")
                    lines.append(f"- **Latest Training:** {latest['timestamp']}")
                    lines.append(f"- **Git SHA:** `{latest['git_sha']}`")
                    lines.append(f"- **Run Number:** {latest['run_number']}")
                    lines.append(f"- **Training Time:** {latest['training_time_seconds']:.2f}s")
                    lines.append("")
                    
                    # Metrics
                    lines.append("#### Metrics")
                    lines.append("")
                    for metric_name, metric_value in latest["metrics"].items():
                        if isinstance(metric_value, float):
                            lines.append(f"- **{metric_name}:** {metric_value:.4f}")
                        else:
                            lines.append(f"- **{metric_name}:** {metric_value}")
                    lines.append("")
                    
                    # Data Info
                    lines.append("#### Training Data")
                    lines.append("")
                    for key, value in latest["data_info"].items():
                        lines.append(f"- **{key}:** {value}")
                    lines.append("")
                    
                    # Comparison with previous
                    if len(model_trainings) >= 2:
                        comparison = self.compare_with_previous(model_type, model_name)
                        if comparison["comparison_available"]:
                            lines.append("#### Comparison with Previous Training")
                            lines.append("")
                            for metric_name, change_info in comparison["metric_changes"].items():
                                emoji = "📈" if change_info["improved"] else "📉"
                                lines.append(
                                    f"- {emoji} **{metric_name}:** "
                                    f"{change_info['previous']:.4f} → {change_info['latest']:.4f} "
                                    f"({change_info['change']:+.4f}, {change_info['change_pct']:+.2f}%)"
                                )
                            lines.append("")
                    
                    lines.append("---")
                    lines.append("")
        
        # Join all lines
        report = "\n".join(lines)
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✅ Generated summary report: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
        
        return report
    
    def export_to_csv(self, output_file: Optional[str] = None) -> str:
        """
        Export training history to CSV format for analysis.
        
        Args:
            output_file: Optional path to save the CSV (default: logs/performance/performance_history.csv)
        
        Returns:
            Path to the saved CSV file
        """
        if output_file is None:
            output_file = self.csv_file
        else:
            output_file = Path(output_file)
        
        if not self.history["trainings"]:
            logger.warning("No training sessions to export")
            return str(output_file)
        
        # Prepare rows
        rows = []
        for training in self.history["trainings"]:
            # Flatten the nested structure
            row = {
                "timestamp": training["timestamp"],
                "model_type": training["model_type"],
                "model_name": training["model_name"],
                "git_sha": training["git_sha"],
                "run_number": training["run_number"],
                "training_time_seconds": training["training_time_seconds"]
            }
            
            # Add metrics
            for metric_name, metric_value in training["metrics"].items():
                row[f"metric_{metric_name}"] = metric_value
            
            # Add data info
            for key, value in training["data_info"].items():
                row[f"data_{key}"] = value
            
            rows.append(row)
        
        # Write to CSV
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info(f"✅ Exported to CSV: {output_file} ({len(rows)} rows)")
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
        
        return str(output_file)
    
    def get_latest_training(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest training record for a specific model.
        
        Args:
            model_type: Type of model
            model_name: Specific model name
        
        Returns:
            Latest training record or None if not found
        """
        model_trainings = [
            t for t in self.history["trainings"]
            if t["model_type"] == model_type and t["model_name"] == model_name
        ]
        
        return model_trainings[-1] if model_trainings else None
    
    def get_all_trainings(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all training records, optionally filtered by model type.
        
        Args:
            model_type: Optional filter by model type
        
        Returns:
            List of training records
        """
        if model_type:
            return [t for t in self.history["trainings"] if t["model_type"] == model_type]
        return self.history["trainings"]
