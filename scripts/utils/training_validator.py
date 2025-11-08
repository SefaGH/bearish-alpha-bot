"""
Training Configuration Validator.

Validates training configuration before starting to prevent
common issues like parameter mismatches, missing values, etc.

Author: SefaGH & GitHub Copilot
Date: 2025-11-08
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class TrainingConfigValidator:
    """Validates training configuration for consistency and completeness."""
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate training configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        warnings = []
        
        # =====================================================================
        # CRITICAL CHECKS (Must pass)
        # =====================================================================
        
        # Check ML enabled
        if not config.get('ml', {}).get('enabled', False):
            issues.append("CRITICAL: ML is disabled (ml.enabled=false)")
        
        # Check training symbols
        symbols = config.get('universe', {}).get('fixed_symbols', [])
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        
        if not symbols:
            issues.append("CRITICAL: No training symbols defined (universe.fixed_symbols is empty)")
        else:
            logger.info(f"Training symbols validated: {len(symbols)} symbols")
        
        # =====================================================================
        # WARNING CHECKS (Can proceed with warnings)
        # =====================================================================
        
        # Check RL training mode
        rl_config = config.get('ml', {}).get('reinforcement_learning', {})
        if not rl_config.get('enabled', False):
            warnings.append("RL is disabled (will skip RL training)")
        elif not rl_config.get('training_mode', False):
            warnings.append("RL training_mode is False (will be forced to True)")
        
        # Check RL epsilon parameters
        required_epsilon_params = ['epsilon_start', 'epsilon_decay', 'epsilon_min']
        missing_epsilon = []
        for param in required_epsilon_params:
            if param not in rl_config:
                missing_epsilon.append(param)
        
        if missing_epsilon:
            warnings.append(f"RL missing epsilon params: {', '.join(missing_epsilon)} (using defaults)")
        
        # Check LSTM parameter consistency
        regime_config = config.get('ml', {}).get('regime_prediction', {})
        lstm_config = regime_config.get('model_params', {}).get('lstm_regime', {})
        
        if not lstm_config:
            warnings.append("LSTM config missing (using code defaults)")
        else:
            required_lstm_params = ['hidden_size', 'num_layers', 'dropout']
            missing_lstm = [p for p in required_lstm_params if p not in lstm_config]
            
            if missing_lstm:
                warnings.append(f"LSTM missing params: {', '.join(missing_lstm)}")
            else:
                # Validate reasonable ranges
                hidden_size = lstm_config.get('hidden_size', 0)
                if not (32 <= hidden_size <= 256):
                    warnings.append(f"LSTM hidden_size={hidden_size} outside typical range [32-256]")
                
                num_layers = lstm_config.get('num_layers', 0)
                if not (1 <= num_layers <= 4):
                    warnings.append(f"LSTM num_layers={num_layers} outside typical range [1-4]")
                
                dropout = lstm_config.get('dropout', 0)
                if not (0.0 <= dropout <= 0.8):
                    warnings.append(f"LSTM dropout={dropout} outside typical range [0.0-0.8]")
        
        # Check timeframes
        timeframes = config.get('ml', {}).get('price_prediction', {}).get('timeframes', [])
        if not timeframes:
            warnings.append("Price prediction timeframes not defined (using defaults)")
        
        is_valid = len(issues) == 0
        all_issues = issues + warnings
        
        return is_valid, all_issues
    
    @staticmethod
    def log_validation_results(is_valid: bool, issues: List[str]) -> None:
        """Log validation results with appropriate severity levels."""
        logger.info("="*70)
        logger.info("🔍 TRAINING CONFIGURATION VALIDATION")
        logger.info("="*70)
        
        if is_valid:
            if not issues:
                logger.info("✅ Configuration is PERFECT - ready for training")
            else:
                logger.info(f"✅ Configuration is VALID with {len(issues)} warning(s)")
                for i, issue in enumerate(issues, 1):
                    if issue.startswith("CRITICAL:"):
                        logger.error(f"   {i}. {issue}")
                    else:
                        logger.warning(f"   {i}. {issue}")
        else:
            logger.error(f"❌ Configuration is INVALID - found {len(issues)} issue(s)")
            for i, issue in enumerate(issues, 1):
                if issue.startswith("CRITICAL:"):
                    logger.error(f"   {i}. {issue}")
                else:
                    logger.warning(f"   {i}. {issue}")
        
        logger.info("="*70)
    
    @staticmethod
    def validate_model_params_sync(config: Dict[str, Any]) -> List[str]:
        """
        Validate that model parameters in config match code constants.
        
        Returns:
            List of synchronization issues (empty if all synced)
        """
        sync_issues = []
        
        try:
            from src.ml import model_trainer
            
            lstm_config = config.get('ml', {}).get('regime_prediction', {}) \
                               .get('model_params', {}).get('lstm_regime', {})
            
            if lstm_config:
                config_hidden = lstm_config.get('hidden_size')
                code_hidden = model_trainer.LSTM_HIDDEN_SIZE
                if config_hidden != code_hidden:
                    sync_issues.append(
                        f"LSTM hidden_size mismatch: config={config_hidden}, "
                        f"model_trainer.py={code_hidden}"
                    )
                
                config_layers = lstm_config.get('num_layers')
                code_layers = model_trainer.LSTM_NUM_LAYERS
                if config_layers != code_layers:
                    sync_issues.append(
                        f"LSTM num_layers mismatch: config={config_layers}, "
                        f"model_trainer.py={code_layers}"
                    )
                
                config_dropout = lstm_config.get('dropout')
                code_dropout = model_trainer.LSTM_DROPOUT
                if config_dropout != code_dropout:
                    sync_issues.append(
                        f"LSTM dropout mismatch: config={config_dropout}, "
                        f"model_trainer.py={code_dropout}"
                    )
        
        except ImportError:
            sync_issues.append("Cannot import model_trainer for sync validation")
        
        return sync_issues
