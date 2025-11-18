import json
import math
from pathlib import Path

import pytest

try:  # pragma: no cover
    from src.ml.reinforcement_learning import TradingRLAgent
except Exception:  # pragma: no cover
    from ml.reinforcement_learning import TradingRLAgent  # type: ignore[import-not-found]

LEGACY_CKPT = Path("tests/data/legacy_rl_agent_head_scale.pth")
LEGACY_META = Path("tests/data/legacy_rl_agent_head_scale_meta.json")


@pytest.mark.skipif(not LEGACY_CKPT.exists(), reason="Legacy RL checkpoint not available")
def test_head_scale_migrates_close():
    agent = TradingRLAgent(state_size=87, config={"head_scale_learnable": True})
    agent.load_model(str(LEGACY_CKPT))

    resolved_scale = agent.get_head_scale_value()

    if LEGACY_META.exists():
        meta = json.loads(LEGACY_META.read_text())
        legacy_scale = float(meta.get("effective_head_scale", resolved_scale))
        assert math.isclose(resolved_scale, legacy_scale, rel_tol=1e-6, abs_tol=1e-6)
    else:
        assert resolved_scale >= agent.head_scale_min_multiplier
