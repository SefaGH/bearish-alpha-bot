import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.reinforcement_learning import TradingRLAgent


def test_full_pipeline(seed: int = 42) -> bool:
    """Kick the full RL pipeline with random states and inspect variance."""
    agent = TradingRLAgent(state_size=82, action_size=3)
    model_path = "data/models/rl_agent_final.pth"
    try:
        agent.load_model(model_path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"❌ Unable to load RL model from {model_path}: {exc}")
        return False

    if hasattr(agent, 'set_inference_mode'):
        agent.set_inference_mode()

    rng = np.random.default_rng(seed)
    states = [rng.standard_normal(82) for _ in range(5)]

    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    q_values = []
    for idx, state in enumerate(states, 1):
        action, meta = agent.get_action_with_meta(
            state,
            market_regime=cast(Any, {'predicted_regime': 'neutral', 'confidence': 0.5}),
        )
        raw_q = meta.get('raw_q_values', [])
        adj_q = meta.get('adjusted_q_values', [])
        q_values.append(raw_q)

        print(f"State {idx}:")
        print(f"  Raw Q: {raw_q}")
        print(f"  Adj Q: {adj_q}")
        print(f"  Action: {['BUY', 'HOLD', 'SELL'][action]}")
        print("-" * 40)

    q_array = np.array(q_values)
    q_std = q_array.std(axis=0)
    mean_std = float(q_std.mean())

    print("Q-value standard deviations:", q_std)
    print(f"Mean std: {mean_std:.6f}")

    if mean_std < 1e-3:
        print("❌ FAIL: Q-values show negligible variance across sampled states")
        return False

    print("✅ PASS: Q-values vary across states")
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    raise SystemExit(0 if success else 1)
