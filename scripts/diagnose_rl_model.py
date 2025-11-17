"""RL Model Diagnostic Tool.

Evaluates scaler outputs, model forward variance, and Q-value differentiation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "data/models/rl_agent_final.pth"
SCALER_PATH = REPO_ROOT / "artifacts/gemma/final/gemma_price_scaler.joblib"

from src.ml.reinforcement_learning import TradingRLAgent  # noqa: E402


def load_scaler(path: Path):
    try:
        scaler = joblib.load(path)
        print(f"✅ Scaler loaded from {path}")
        return scaler
    except FileNotFoundError:
        print(f"⚠️ No scaler found at {path} (continuing with raw features)")
        return None
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Failed to load scaler {path}: {exc}")
        return None


def load_agent() -> TradingRLAgent:
    agent = TradingRLAgent(state_size=82, action_size=3)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    agent.load_model(str(MODEL_PATH))
    model = getattr(agent, "q_network", getattr(agent, "model", None))
    if model is None:
        raise RuntimeError("RL agent has no underlying model instance (q_network/model missing)")
    model.eval()
    return agent


def run_inference(agent: TradingRLAgent, state_vec: np.ndarray) -> np.ndarray:
    model = getattr(agent, "q_network", getattr(agent, "model", None))
    if model is None:
        raise RuntimeError("RL agent lacks a model for inference")
    with torch.no_grad():
        tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        return model(tensor).cpu().numpy()[0]


def pretty(values: np.ndarray) -> List[float]:
    return np.round(values, 6).tolist()


def build_test_cases(rng_seed: int = 42) -> List[Tuple[str, np.ndarray]]:
    log_state1 = np.array(
        [9.45666000e04, 9.45625392e04, 2.37795000e01, 4.95402000e01, 5.01690000e01]
        + [0.0] * 77
    )
    log_state2 = np.array(
        [9.42614600e04, 9.42559169e04, 0.0, 7.12776000e01, 7.40257000e01]
        + [0.0] * 77
    )
    cases: List[Tuple[str, np.ndarray]] = [
        ("log_15_12_57", log_state1),
        ("log_14_21_45", log_state2),
    ]
    rng = np.random.default_rng(rng_seed)
    for idx in range(5):
        cases.append((f"random_{idx+1}", rng.standard_normal(82)))
    return cases


def main() -> None:
    print("=" * 60)
    print("RL MODEL DIAGNOSTIC")
    print("=" * 60)

    print("\n[STEP 1] Loading RL agent and scaler ...")
    agent = load_agent()
    scaler = load_scaler(SCALER_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   training_mode={agent.training_mode}")
    print(f"   epsilon={agent.epsilon}")

    print("\n[STEP 2] Preparing test cases ...")
    cases = build_test_cases()
    print(f"✅ Prepared {len(cases)} cases")

    print("\n[STEP 3] Running inference tests ...")
    print("=" * 60)

    diagnostics: List[Dict[str, float]] = []
    for name, raw_state in cases:
        scaled_state = raw_state
        if scaler is not None:
            try:
                scaled_state = scaler.transform(raw_state.reshape(1, -1))[0]
            except Exception as exc:
                print(f"⚠️ [{name}] Scaler transform failed: {exc}")
        q_values = run_inference(agent, scaled_state)
        q_std = float(np.std(q_values))
        q_range = float(np.max(q_values) - np.min(q_values))
        diagnostics.append({
            'name': name,
            'q_std': q_std,
            'q_range': q_range,
        })
        print(f"\n[{name}]")
        print(f"  Raw (first 5):    {pretty(raw_state[:5])}")
        print(f"  Scaled (first 5): {pretty(scaled_state[:5])}")
        print(f"  Q-values:         {pretty(q_values)}")
        print(f"  Q std:            {q_std:.8f}")
        print(f"  Q range:          {q_range:.8f}")
        print(f"  Decision:         {['BUY','HOLD','SELL'][int(np.argmax(q_values))]}")

    print("\n" + "=" * 60)
    print("[STEP 4] Pairwise comparison of log samples")
    print("=" * 60)
    log_a = diagnostics[0]
    log_b = diagnostics[1]
    diff = abs(log_a['q_std'] - log_b['q_std'])
    print(f"Log sample stds: {log_a['q_std']:.8f} vs {log_b['q_std']:.8f} (diff={diff:.8f})")

    print("\n" + "=" * 60)
    print("[STEP 5] Statistical summary")
    print("=" * 60)
    q_stds = np.array([d['q_std'] for d in diagnostics])
    q_ranges = np.array([d['q_range'] for d in diagnostics])
    print(f"Q-std min/median/max: {q_stds.min():.8f} / {np.median(q_stds):.8f} / {q_stds.max():.8f}")
    print(f"Q-range min/median/max: {q_ranges.min():.8f} / {np.median(q_ranges):.8f} / {q_ranges.max():.8f}")

    print("\n" + "=" * 60)
    print("[STEP 6] Diagnosis")
    print("=" * 60)
    median_std = float(np.median(q_stds))
    if median_std < 1e-4:
        print("❌ MODEL FROZEN: Q-values have negligible variance")
        print("   → Re-export or retrain the RL model")
    elif median_std < 1e-3:
        print("⚠️ LOW VARIANCE: Model sensitivity is weak")
        print("   → Investigate scaler / feature pipeline")
    else:
        print("✅ VARIANCE OK: Model reacts to inputs")

    max_diff = float(np.max(np.abs(q_stds[:, None] - q_stds)))
    if max_diff < 1e-4:
        print("❌ STATES NOT DIFFERENTIATED: Distinct inputs produce identical Q-values")
    elif max_diff < 1e-3:
        print("⚠️ WEAK DIFFERENTIATION: Differences are minimal")
    else:
        print("✅ DIFFERENTIATION OK: Inputs produce distinct Q-values")

    if scaler is None:
        print("\n(Scaler unavailable – skipping scaler diagnostics)")


if __name__ == "__main__":
    main()
