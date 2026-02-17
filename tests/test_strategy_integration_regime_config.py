from unittest.mock import MagicMock

import pytest

from ml.strategy_integration import AIEnhancedStrategyAdapter


def test_adapter_reads_soft_weight_thresholds_from_regime_prediction_block():
    adapter = AIEnhancedStrategyAdapter(
        price_engine=MagicMock(),
        regime_predictor=MagicMock(),
        config={
            "regime_prediction": {
                "min_confidence_hard_reject": 0.40,
                "min_confidence_full_weight": 0.80,
            }
        },
    )

    assert adapter.regime_hard_reject == pytest.approx(0.40)
    assert adapter.regime_full_weight == pytest.approx(0.80)
    assert adapter._calculate_regime_weight(0.35) is None
    assert adapter._calculate_regime_weight(0.60) == pytest.approx(0.75)

