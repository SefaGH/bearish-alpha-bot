import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.reinforcement_learning import TradingRLAgent


def test_model_forward() -> bool:
    """Ensure RL model reacts to different state inputs."""
    agent = TradingRLAgent(state_size=82, action_size=3)
    model_path = "data/models/rl_agent_final.pth"
    try:
        agent.load_model(model_path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"❌ Unable to load RL model from {model_path}: {exc}")
        return False

    model = getattr(agent, 'q_network', getattr(agent, 'model', None))
    if model is None:
        print("❌ RL agent has no underlying model instance")
        return False

    model.eval()

    state1 = np.array(
        [9.42614600e04, 9.42559169e04, 0.0, 7.12776000e01, 7.40257000e01] + [0.0] * 77
    )
    state2 = np.array(
        [9.40721800e04, 9.40745904e04, 9.901e-01, 4.94091000e01, 4.94184000e01] + [0.0] * 77
    )

    with torch.no_grad():
        tensor1 = torch.FloatTensor(state1).unsqueeze(0)
        tensor2 = torch.FloatTensor(state2).unsqueeze(0)
        q1 = model(tensor1).cpu().numpy()[0]
        q2 = model(tensor2).cpu().numpy()[0]

    diff = np.abs(q1 - q2)
    max_diff = float(np.max(diff))

    print("=" * 60)
    print("MODEL FORWARD TEST")
    print("=" * 60)
    print(f"State 1 Q-values: {q1}")
    print(f"State 2 Q-values: {q2}")
    print(f"Difference:       {diff}")
    print(f"Max difference:   {max_diff}")

    if max_diff < 1e-3:
        print("❌ FAIL: Model produces near-identical outputs")
        return False

    print("✅ PASS: Model responds differently to distinct states")
    return True


if __name__ == "__main__":
    success = test_model_forward()
    raise SystemExit(0 if success else 1)
