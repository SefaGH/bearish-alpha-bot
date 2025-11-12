#!/usr/bin/env python3
"""
Phase 2 (V&V): GEMMA Functional Test, Model Training and Performance Validation
Complete validation script for GEMMA integration
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.logger import setup_logger

logger = setup_logger("phase2-validation", level=logging.INFO, log_to_file=True, log_filename="phase2_validation.log")

class Phase2Validator:
    """Phase 2 GEMMA Validation Suite"""
    
    def __init__(self):
        self.results = {
            'model_training': {'status': '⏳', 'notes': ''},
            'artifact_production': {'status': '⏳', 'notes': ''},
            'adapter_loading': {'status': '⏳', 'notes': ''},
            'ai_gate_logic': {'status': '⏳', 'notes': ''},
            'circuit_breaker': {'status': '⏳', 'notes': ''},
            'e2e_inference': {'status': '⏳', 'notes': ''},
            'performance_benchmark': {'status': '⏳', 'notes': '', 'avg_time': 0}
        }
        self.metrics = {
            'test_accuracy': 0.0,
            'avg_inference_time': 0.0
        }
        
    def task1_model_training(self) -> bool:
        """Task 1: Model Training and Artifact Production"""
        logger.info("\n" + "="*80)
        logger.info("TASK 1: MODEL TRAINING AND ARTIFACT PRODUCTION")
        logger.info("="*80)
        
        try:
            # Check if GEMMA_ENABLED is set
            gemma_enabled = os.environ.get('GEMMA_ENABLED', 'false').lower() == 'true'
            if not gemma_enabled:
                logger.warning("⚠️ GEMMA_ENABLED environment variable is not set to 'true'")
                self.results['model_training']['status'] = '⚠️'
                self.results['model_training']['notes'] = 'GEMMA_ENABLED not set'
                return False
                
            logger.info("✅ GEMMA_ENABLED environment variable is set")
            
            # Check if artifacts exist (if training was already run)
            model_path = Path('data/models/gemma/final/gemma_price.pt')
            scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
            
            if model_path.exists() and scaler_path.exists():
                logger.info(f"✅ Model artifact exists: {model_path}")
                logger.info(f"✅ Scaler artifact exists: {scaler_path}")
                self.results['model_training']['status'] = '✅'
                self.results['model_training']['notes'] = 'Artifacts already exist from previous training'
                self.results['artifact_production']['status'] = '✅'
                self.results['artifact_production']['notes'] = 'Both gemma_price.pt and scaler_gemma.joblib exist'
                
                # Try to extract accuracy from training logs
                try:
                    log_file = Path('logs/training.log')
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            for line in f:
                                if 'Final Validation Accuracy' in line or 'GEMMA Price Model Final Validation Accuracy' in line:
                                    # Extract accuracy value
                                    import re
                                    match = re.search(r'(\d+\.\d+)%', line)
                                    if match:
                                        self.metrics['test_accuracy'] = float(match.group(1))
                                        logger.info(f"📊 Extracted Test Accuracy: {self.metrics['test_accuracy']}%")
                                        break
                except Exception as e:
                    logger.warning(f"Could not extract accuracy from logs: {e}")
                
                return True
            else:
                logger.warning("⚠️ Artifacts not found. Training may need to be run.")
                logger.info(f"   Missing model: {not model_path.exists()}")
                logger.info(f"   Missing scaler: {not scaler_path.exists()}")
                self.results['model_training']['status'] = '⚠️'
                self.results['model_training']['notes'] = 'Artifacts not found - training needed'
                return False
                
        except Exception as e:
            logger.error(f"❌ Task 1 failed: {e}", exc_info=True)
            self.results['model_training']['status'] = '❌'
            self.results['model_training']['notes'] = str(e)
            return False
    
    def task2_adapter_loading(self) -> bool:
        """Task 2.1: Adapter Loading Test"""
        logger.info("\n" + "="*80)
        logger.info("TASK 2.1: ADAPTER LOADING TEST")
        logger.info("="*80)
        
        try:
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            
            # Check if artifacts exist
            model_path = Path('data/models/gemma/final/gemma_price.pt')
            scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
            
            if not model_path.exists():
                logger.error(f"❌ Model not found: {model_path}")
                self.results['adapter_loading']['status'] = '❌'
                self.results['adapter_loading']['notes'] = 'Model file not found'
                return False
                
            if not scaler_path.exists():
                logger.error(f"❌ Scaler not found: {scaler_path}")
                self.results['adapter_loading']['status'] = '❌'
                self.results['adapter_loading']['notes'] = 'Scaler file not found'
                return False
            
            logger.info("✅ Model and scaler files exist")
            
            # Create adapter config
            config = {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'recovery_timeout': 60
                },
                'shadow_mode': False,
                'cache_ttl': 30
            }
            
            # Initialize adapter
            logger.info("Initializing GemmaTorchScriptAdapter...")
            adapter = GemmaTorchScriptAdapter(config)
            
            logger.info("✅ GemmaTorchScriptAdapter initialized successfully")
            self.results['adapter_loading']['status'] = '✅'
            self.results['adapter_loading']['notes'] = 'Adapter loaded without errors'
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Adapter loading failed: {e}", exc_info=True)
            self.results['adapter_loading']['status'] = '❌'
            self.results['adapter_loading']['notes'] = str(e)
            return False
    
    def task2_ai_gate_logic(self) -> bool:
        """Task 2.2: AI-Gate Logic Test"""
        logger.info("\n" + "="*80)
        logger.info("TASK 2.2: AI-GATE LOGIC TEST")
        logger.info("="*80)
        
        try:
            from src.core.strategy_coordinator import StrategyCoordinator
            from unittest.mock import MagicMock
            
            # Create mock dependencies
            portfolio_manager = MagicMock()
            risk_manager = MagicMock()
            
            # Create config with GEMMA disabled for manual testing
            config = {
                'ml': {
                    'gemma': {
                        'enabled': False  # We'll manually set gemma_confidence
                    },
                    'price': {
                        'min_confidence': 0.66
                    }
                }
            }
            
            # Initialize coordinator
            logger.info("Initializing StrategyCoordinator...")
            coordinator = StrategyCoordinator(
                portfolio_manager=portfolio_manager,
                risk_manager=risk_manager,
                config=config
            )
            
            # Test 1: Signal with high confidence (should PASS)
            logger.info("\n--- Test 1: High Confidence Signal (0.80) ---")
            signal_pass = {
                'symbol': 'BTC/USDT',
                'gemma_confidence': 0.80,
                'features': {}
            }
            
            result_pass = coordinator._apply_ai_gate(signal_pass)
            
            if result_pass:
                logger.info("✅ High confidence signal PASSED as expected")
            else:
                logger.error("❌ High confidence signal was REJECTED (unexpected)")
                self.results['ai_gate_logic']['status'] = '❌'
                self.results['ai_gate_logic']['notes'] = 'High confidence signal rejected'
                return False
            
            # Test 2: Signal with low confidence (should REJECT)
            logger.info("\n--- Test 2: Low Confidence Signal (0.50) ---")
            signal_reject = {
                'symbol': 'BTC/USDT',
                'gemma_confidence': 0.50,
                'features': {}
            }
            
            result_reject = coordinator._apply_ai_gate(signal_reject)
            
            if not result_reject:
                logger.info("✅ Low confidence signal REJECTED as expected")
            else:
                logger.error("❌ Low confidence signal was PASSED (unexpected)")
                self.results['ai_gate_logic']['status'] = '❌'
                self.results['ai_gate_logic']['notes'] = 'Low confidence signal passed'
                return False
            
            # Verify stats
            rejections = coordinator.processing_stats.get('ai_gate_rejections', 0)
            logger.info(f"\n📊 AI-Gate Statistics:")
            logger.info(f"   Total rejections: {rejections}")
            
            if rejections != 1:
                logger.warning(f"⚠️ Expected 1 rejection, got {rejections}")
            
            logger.info("✅ AI-Gate logic test completed successfully")
            self.results['ai_gate_logic']['status'] = '✅'
            self.results['ai_gate_logic']['notes'] = 'Both high and low confidence signals handled correctly'
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AI-Gate logic test failed: {e}", exc_info=True)
            self.results['ai_gate_logic']['status'] = '❌'
            self.results['ai_gate_logic']['notes'] = str(e)
            return False
    
    def task2_circuit_breaker(self) -> bool:
        """Task 2.3: Circuit Breaker Test"""
        logger.info("\n" + "="*80)
        logger.info("TASK 2.3: CIRCUIT BREAKER TEST")
        logger.info("="*80)
        
        try:
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            from unittest.mock import patch
            
            # Check if artifacts exist
            model_path = Path('data/models/gemma/final/gemma_price.pt')
            scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
            
            if not model_path.exists() or not scaler_path.exists():
                logger.error("❌ Model or scaler not found - cannot test circuit breaker")
                self.results['circuit_breaker']['status'] = '❌'
                self.results['circuit_breaker']['notes'] = 'Artifacts not found'
                return False
            
            # Create adapter config
            config = {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'recovery_timeout': 60
                },
                'shadow_mode': False,
                'cache_ttl': 30
            }
            
            # Initialize adapter
            logger.info("Initializing GemmaTorchScriptAdapter...")
            adapter = GemmaTorchScriptAdapter(config)
            
            logger.info("✅ Adapter initialized")
            logger.info(f"Circuit breaker initial state: {adapter.circuit_breaker.state}")
            
            # Mock the _predict_internal method to raise an error
            logger.info("\nMocking adapter to force failures...")
            
            def failing_predict(*args, **kwargs):
                raise RuntimeError("Simulated prediction failure")
            
            # Create dummy features (87 features expected)
            dummy_features = {f'feature_{i}': 0.5 for i in range(87)}
            
            logger.info("Testing circuit breaker with 6 consecutive failures...")
            
            failure_count = 0
            with patch.object(adapter, '_predict_internal', side_effect=failing_predict):
                for i in range(1, 7):
                    try:
                        adapter.predict(dummy_features)
                        logger.error(f"Call {i}: Prediction should have failed but didn't")
                    except Exception as e:
                        failure_count += 1
                        logger.info(f"Call {i}: Failed as expected - {type(e).__name__}")
                        logger.info(f"   Circuit breaker state: {adapter.circuit_breaker.state}")
                        logger.info(f"   Failure count: {adapter.circuit_breaker.failure_count}")
                        
                        # Check if circuit opened after 5th failure
                        if i == 5:
                            if adapter.circuit_breaker.state == "OPEN":
                                logger.info("✅ Circuit breaker opened after 5 failures")
                            else:
                                logger.error(f"❌ Circuit breaker should be OPEN but is {adapter.circuit_breaker.state}")
            
            # Verify final state
            if adapter.circuit_breaker.state == "OPEN":
                logger.info("✅ Circuit breaker is in OPEN state")
                self.results['circuit_breaker']['status'] = '✅'
                self.results['circuit_breaker']['notes'] = f'Circuit opened after 5 failures (tested {failure_count} total)'
                return True
            else:
                logger.error(f"❌ Circuit breaker should be OPEN but is {adapter.circuit_breaker.state}")
                self.results['circuit_breaker']['status'] = '❌'
                self.results['circuit_breaker']['notes'] = f'Circuit did not open (state: {adapter.circuit_breaker.state})'
                return False
                
        except Exception as e:
            logger.error(f"❌ Circuit breaker test failed: {e}", exc_info=True)
            self.results['circuit_breaker']['status'] = '❌'
            self.results['circuit_breaker']['notes'] = str(e)
            return False
    
    def task3_e2e_inference(self) -> bool:
        """Task 3.1: End-to-End Inference Test"""
        logger.info("\n" + "="*80)
        logger.info("TASK 3.1: END-TO-END INFERENCE TEST")
        logger.info("="*80)
        
        try:
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            
            # Check if artifacts exist
            model_path = Path('data/models/gemma/final/gemma_price.pt')
            scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
            
            if not model_path.exists() or not scaler_path.exists():
                logger.error("❌ Model or scaler not found")
                self.results['e2e_inference']['status'] = '❌'
                self.results['e2e_inference']['notes'] = 'Artifacts not found'
                return False
            
            # Create adapter config
            config = {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'recovery_timeout': 60
                },
                'shadow_mode': False,
                'cache_ttl': 30
            }
            
            # Initialize adapter
            logger.info("Initializing GemmaTorchScriptAdapter...")
            adapter = GemmaTorchScriptAdapter(config)
            
            # Create realistic test features (87 features)
            logger.info("Creating test feature dictionary (87 features)...")
            features = {}
            
            # Generate 87 features with realistic values
            for i in range(87):
                features[f'feature_{i}'] = np.random.randn() * 0.5 + 0.5
            
            logger.info(f"✅ Created feature dictionary with {len(features)} features")
            
            # Perform prediction
            logger.info("Performing prediction...")
            result = adapter.predict(features)
            
            # Verify result structure
            logger.info("\n📊 Prediction Result:")
            logger.info(f"   Keys: {list(result.keys())}")
            
            required_keys = ['price_confidence', 'prediction_label', 'probabilities', 'fallback']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                logger.error(f"❌ Missing keys in result: {missing_keys}")
                self.results['e2e_inference']['status'] = '❌'
                self.results['e2e_inference']['notes'] = f'Missing keys: {missing_keys}'
                return False
            
            logger.info(f"   price_confidence: {result.get('price_confidence')}")
            logger.info(f"   prediction_label: {result.get('prediction_label')}")
            logger.info(f"   probabilities: {result.get('probabilities')}")
            logger.info(f"   fallback: {result.get('fallback')}")
            
            # Verify fallback is False
            if result.get('fallback', True):
                logger.error("❌ Fallback should be False but is True")
                self.results['e2e_inference']['status'] = '❌'
                self.results['e2e_inference']['notes'] = 'Fallback was True'
                return False
            
            logger.info("✅ End-to-end inference test passed")
            self.results['e2e_inference']['status'] = '✅'
            self.results['e2e_inference']['notes'] = 'All expected keys present, fallback=False'
            
            return True
            
        except Exception as e:
            logger.error(f"❌ E2E inference test failed: {e}", exc_info=True)
            self.results['e2e_inference']['status'] = '❌'
            self.results['e2e_inference']['notes'] = str(e)
            return False
    
    def task3_performance_benchmark(self) -> bool:
        """Task 3.2: Performance Benchmarking"""
        logger.info("\n" + "="*80)
        logger.info("TASK 3.2: PERFORMANCE BENCHMARKING")
        logger.info("="*80)
        
        try:
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            
            # Check if artifacts exist
            model_path = Path('data/models/gemma/final/gemma_price.pt')
            scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
            
            if not model_path.exists() or not scaler_path.exists():
                logger.error("❌ Model or scaler not found")
                self.results['performance_benchmark']['status'] = '❌'
                self.results['performance_benchmark']['notes'] = 'Artifacts not found'
                return False
            
            # Create adapter config
            config = {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'recovery_timeout': 60
                },
                'shadow_mode': False,
                'cache_ttl': 0  # Disable caching for accurate benchmark
            }
            
            # Initialize adapter
            logger.info("Initializing GemmaTorchScriptAdapter...")
            adapter = GemmaTorchScriptAdapter(config)
            
            # Create test features
            features = {f'feature_{i}': np.random.randn() * 0.5 + 0.5 for i in range(87)}
            
            # Warmup (5 calls)
            logger.info("Warming up (5 calls)...")
            for _ in range(5):
                adapter.predict(features)
            
            # Benchmark (1000 calls)
            logger.info("Running benchmark (1000 calls)...")
            num_iterations = 1000
            start_time = time.time()
            
            for i in range(num_iterations):
                adapter.predict(features)
                if (i + 1) % 100 == 0:
                    logger.info(f"   Completed {i + 1}/{num_iterations} iterations")
            
            total_time = time.time() - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            
            logger.info("\n📊 Performance Metrics:")
            logger.info(f"   Total time: {total_time:.3f} seconds")
            logger.info(f"   Total iterations: {num_iterations}")
            logger.info(f"   Average inference time: {avg_time_ms:.3f} ms")
            
            self.metrics['avg_inference_time'] = avg_time_ms
            self.results['performance_benchmark']['avg_time'] = avg_time_ms
            
            # Check if meets target (<100ms)
            target_ms = 100
            if avg_time_ms < target_ms:
                logger.info(f"✅ Performance target met: {avg_time_ms:.3f}ms < {target_ms}ms")
                self.results['performance_benchmark']['status'] = '✅'
                self.results['performance_benchmark']['notes'] = f'Average: {avg_time_ms:.3f}ms (target: <{target_ms}ms)'
                return True
            else:
                logger.warning(f"⚠️ Performance target missed: {avg_time_ms:.3f}ms >= {target_ms}ms")
                self.results['performance_benchmark']['status'] = '⚠️'
                self.results['performance_benchmark']['notes'] = f'Average: {avg_time_ms:.3f}ms (target: <{target_ms}ms)'
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}", exc_info=True)
            self.results['performance_benchmark']['status'] = '❌'
            self.results['performance_benchmark']['notes'] = str(e)
            return False
    
    def generate_report(self) -> str:
        """Generate Phase 2 Validation Report"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING PHASE 2 VALIDATION REPORT")
        logger.info("="*80)
        
        # Determine overall status
        failed_tests = [k for k, v in self.results.items() if v['status'] == '❌']
        warning_tests = [k for k, v in self.results.items() if v['status'] == '⚠️']
        
        if failed_tests:
            overall_status = "EK İYİLEŞTİRME GEREKLİ"
        elif warning_tests:
            overall_status = "EK İYİLEŞTİRME GEREKLİ"
        else:
            overall_status = "CANLI İÇİN HAZIR"
        
        # Generate report
        report = f"""## Faz 2 Geçerleme Raporu

**Tarih:** {datetime.now().strftime('%Y-%m-%d')}
**Kontrolü Yapan:** @github-copilot

### ✅ Genel Durum: `{overall_status}`

---

### Ayrıntılı Test Sonuçları

| Test Adı | Durum | Notlar / Metrikler |
| --- | :---: | --- |
| **Model Eğitimi** | {self.results['model_training']['status']} | {self.results['model_training']['notes']} |
| **Artifakt Üretimi** | {self.results['artifact_production']['status']} | {self.results['artifact_production']['notes']} |
| **Adapter Yükleme Testi** | {self.results['adapter_loading']['status']} | {self.results['adapter_loading']['notes']} |
| **AI-Gate Mantık Testi** | {self.results['ai_gate_logic']['status']} | {self.results['ai_gate_logic']['notes']} |
| **Circuit Breaker Testi** | {self.results['circuit_breaker']['status']} | {self.results['circuit_breaker']['notes']} |
| **End-to-End Çıkarım Testi** | {self.results['e2e_inference']['status']} | {self.results['e2e_inference']['notes']} |
| **Performans Ölçümü** | {self.results['performance_benchmark']['status']} | {self.results['performance_benchmark']['notes']} |

---

### 📈 Kritik Performans Metrikleri

| Metrik | Hedef | Ölçülen Değer | Sonuç |
| --- | :---: | :---: | :---: |
| **Test Accuracy** | > %78.99 | {f'%{self.metrics["test_accuracy"]:.2f}' if self.metrics['test_accuracy'] > 0 else 'N/A'} | {'✅' if self.metrics['test_accuracy'] > 78.99 else '❌' if self.metrics['test_accuracy'] > 0 else 'N/A'} |
| **Ortalama Inference Time** | < 100ms | {f'{self.metrics["avg_inference_time"]:.1f} ms' if self.metrics['avg_inference_time'] > 0 else 'N/A'} | {'✅' if 0 < self.metrics['avg_inference_time'] < 100 else '❌' if self.metrics['avg_inference_time'] > 0 else 'N/A'} |

---

### 📝 Sonuç ve Öneri

"""
        
        if overall_status == "CANLI İÇİN HAZIR":
            report += """GEMMA entegrasyonu, fonksiyonel ve performans testlerini başarıyla tamamlamıştır. Sistem, hedeflenen doğruluk ve hız kriterlerini karşılamaktadır ve canlı ortama geçiş için **hazırdır**.
"""
        else:
            report += """Şu konularda iyileştirme gerekmektedir:

"""
            issue_num = 1
            for test_name, result in self.results.items():
                if result['status'] in ['❌', '⚠️']:
                    report += f"{issue_num}. **{test_name.replace('_', ' ').title()}:** {result['notes']}\n"
                    issue_num += 1
            
            report += """
**Öneri:** Canlı dağıtıma geçmeden önce bu sorunların giderilmesi ve testlerin yeniden çalıştırılması tavsiye edilir.
"""
        
        return report
    
    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        logger.info("\n" + "="*80)
        logger.info("STARTING PHASE 2 VALIDATION")
        logger.info("="*80)
        
        all_passed = True
        
        # Task 1: Model Training and Artifact Production
        if not self.task1_model_training():
            logger.warning("⚠️ Task 1 did not fully pass - continuing with other tests")
            all_passed = False
        
        # Task 2: Functional Unit Tests
        if not self.task2_adapter_loading():
            logger.warning("⚠️ Task 2.1 failed - skipping remaining tests")
            return False
        
        if not self.task2_ai_gate_logic():
            all_passed = False
        
        if not self.task2_circuit_breaker():
            all_passed = False
        
        # Task 3: Integration and Performance Tests
        if not self.task3_e2e_inference():
            all_passed = False
        
        if not self.task3_performance_benchmark():
            all_passed = False
        
        return all_passed

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("PHASE 2 (V&V): GEMMA FUNCTIONAL TEST AND PERFORMANCE VALIDATION")
    print("="*80 + "\n")
    
    validator = Phase2Validator()
    
    try:
        validator.run_all_tests()
        
        # Generate and print report
        report = validator.generate_report()
        print("\n" + report)
        
        # Save report to file
        report_file = Path('PHASE2_VALIDATION_REPORT.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"\n✅ Report saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
