#!/usr/bin/env python3
"""RL Model Diagnostic Tool.

Tests scaler, model forward pass, and Q-value variance.

Usage:
    python scripts/diagnose_rl_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "data/models/rl_agent_final.pth"
SCALER_PATH = REPO_ROOT / "artifacts/gemma/final/gemma_price_scaler.joblib"

from src.ml.reinforcement_learning import TradingRLAgent  # noqa: E402


def _infer_checkpoint_state_size(path: Path) -> int | None:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        raise
    except Exception:  # pragma: no cover - handled by caller for diagnostics
        return None

    state_dict = checkpoint.get("q_network")
    if not isinstance(state_dict, dict):
        return None

    for key, tensor in state_dict.items():
        if key.endswith("network.0.weight") and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[1])
    return None


def _get_underlying_model(agent: TradingRLAgent):
    """Return the underlying torch module regardless of attribute name."""

    return getattr(agent, 'q_network', getattr(agent, 'model', None))


def load_components() -> Tuple[TradingRLAgent, Any]:
    """Load RL agent and scaler from disk."""

    print("Loading RL Agent...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    checkpoint_state = _infer_checkpoint_state_size(MODEL_PATH)
    expected_state = 82
    if checkpoint_state is not None and checkpoint_state != expected_state:
        raise ValueError(
            "Checkpoint state dimension mismatch: "
            f"checkpoint={checkpoint_state}, expected={expected_state}. "
            "Re-export the model or pass a matching checkpoint."
        )

    agent = TradingRLAgent(state_size=82, action_size=3)
    agent.load_model(str(MODEL_PATH))
    model = _get_underlying_model(agent)
    if model is None:
        raise RuntimeError("TradingRLAgent has no underlying model after load_model().")
    model.eval()
    print(f"[OK] Model loaded: {MODEL_PATH}")

    print("\nLoading Scaler...")
    try:
        scaler = joblib.load(str(SCALER_PATH))
        print(f"[OK] Scaler loaded: {SCALER_PATH}")
    except FileNotFoundError:
        print(f"[WARN] Scaler not found at {SCALER_PATH}")
        scaler = None
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"[FAIL] Failed to load scaler: {exc}")
        scaler = None

    return agent, scaler


def run_inference(agent: TradingRLAgent, state_vec: np.ndarray) -> np.ndarray:
    """Run model inference with proper eval mode."""

    model = _get_underlying_model(agent)
    if model is None:
        raise RuntimeError("TradingRLAgent has no model/q_network attribute for inference")
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        q_values = model(tensor).cpu().numpy()[0]
    return q_values


def pretty(values: np.ndarray, decimals: int = 6) -> List[float]:
    """Convert numpy array to rounded list for printing."""

    return np.round(values, decimals).tolist()


def build_test_cases() -> List[Tuple[str, np.ndarray]]:
    """Prepare deterministic log states plus random samples."""

    state_log1 = np.array([
        93155.0, 93152.0, 26.53, 5.39, 5.76
    ] + [0.0] * 77)

    state_log2 = np.array([
        93034.0, 93039.0, 98.18, 3.43, 1.96
    ] + [0.0] * 77)

    rng = np.random.RandomState(42)
    random_states = [rng.randn(82) * 10000 + 93000 for _ in range(5)]

    cases = [
        ("log_16:15:19", state_log1),
        ("log_16:20:20", state_log2),
    ]

    for idx, rs in enumerate(random_states, start=1):
        cases.append((f"random_{idx}", rs))

    return cases


def main() -> None:
    """Run diagnostic pipeline end-to-end."""

    print("=" * 80)
    print("[DIAG] RL MODEL DIAGNOSTIC")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load components
    # ------------------------------------------------------------------
    print("\n[STEP 1/5] Loading components...")
    agent, scaler = load_components()

    print("\nModel Info:")
    print(f"  Training mode: {agent.training_mode}")
    print(f"  Epsilon: {agent.epsilon}")

    # ------------------------------------------------------------------
    # 2. Prepare test cases
    # ------------------------------------------------------------------
    print("\n[STEP 2/5] Preparing test cases...")
    cases = build_test_cases()
    print(f"[OK] Prepared {len(cases)} test cases")

    # ------------------------------------------------------------------
    # 3. Run inference tests
    # ------------------------------------------------------------------
    print("\n[STEP 3/5] Running inference tests...")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    for name, raw_state in cases:
        if scaler is not None:
            try:
                scaled_state = scaler.transform(raw_state.reshape(1, -1))[0]
            except Exception as exc:
                print(f"[WARN] [{name}] Scaler failed: {exc}")
                scaled_state = raw_state
        else:
            scaled_state = raw_state

        q_values = run_inference(agent, scaled_state)
        q_std = float(np.std(q_values))
        q_range = float(np.max(q_values) - np.min(q_values))

        results.append({
            'name': name,
            'raw': raw_state,
            'scaled': scaled_state,
            'q_values': q_values,
            'q_std': q_std,
            'q_range': q_range,
        })

        print(f"\n[{name}]")
        print(f"  Raw (first 5):    {pretty(raw_state[:5], 2)}")
        print(f"  Scaled (first 5): {pretty(scaled_state[:5], 4)}")
        print(f"  Q-values:         {pretty(q_values, 6)}")
        print(f"  Q-std:            {q_std:.8f}")
        print(f"  Q-range:          {q_range:.8f}")
        print(f"  Decision:         {['BUY', 'HOLD', 'SELL'][int(np.argmax(q_values))]}")

    # ------------------------------------------------------------------
    # 4. Pairwise comparison of log samples
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 4/5] Pairwise Comparison")
    print("=" * 80)

    log1_q = results[0]['q_values']
    log2_q = results[1]['q_values']
    q_diff = np.abs(log1_q - log2_q)
    max_diff = float(np.max(q_diff))

    print(f"\nLog 16:15:19 Q-values: {pretty(log1_q)}")
    print(f"Log 16:20:20 Q-values: {pretty(log2_q)}")
    print(f"Absolute difference:   {pretty(q_diff)}")
    print(f"Max difference:        {max_diff:.8f}")

    # ------------------------------------------------------------------
    # 5. Statistical analysis & diagnosis
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 5/5] Statistical Analysis")
    print("=" * 80)

    all_q_stds = np.array([r['q_std'] for r in results])
    all_q_ranges = np.array([r['q_range'] for r in results])

    print(f"\nQ-value standard deviations:")
    print(f"  Min:    {all_q_stds.min():.8f}")
    print(f"  Max:    {all_q_stds.max():.8f}")
    print(f"  Mean:   {all_q_stds.mean():.8f}")
    print(f"  Median: {np.median(all_q_stds):.8f}")

    print(f"\nQ-value ranges:")
    print(f"  Min:    {all_q_ranges.min():.8f}")
    print(f"  Max:    {all_q_ranges.max():.8f}")
    print(f"  Mean:   {all_q_ranges.mean():.8f}")
    print(f"  Median: {np.median(all_q_ranges):.8f}")

    print("\n" + "=" * 80)
    print("[REPORT] DIAGNOSIS")
    print("=" * 80)

    median_std = float(np.median(all_q_stds))
    median_range = float(np.median(all_q_ranges))

    print("\n[CHECK] Diagnostic Results:")

    if median_std < 0.0001:
        print("[FAIL] MODEL FROZEN: Q-values have no variance")
        print("   -> Recommendation: Re-export model or check training")
        diagnosis = "FROZEN"
    elif median_std < 0.001:
        print("[WARN] LOW VARIANCE: Q-values are very similar")
        print("   -> Recommendation: Implement fallback logic + check features")
        diagnosis = "LOW_VARIANCE"
    else:
        print("[OK] VARIANCE OK: Model produces varied outputs")
        diagnosis = "OK"

    print(f"\nQ-range median: {median_range:.8f}")

    max_diff_std = float(np.max(np.abs(all_q_stds[:, None] - all_q_stds))) if len(all_q_stds) else 0.0

    if max_diff < 0.0001:
        print("[FAIL] STATES NOT DIFFERENTIATED: Different inputs -> Same Q-values")
        print("   Possible causes: scaler identical outputs or NaN handling")
    elif max_diff < 0.001:
        print("[WARN] WEAK DIFFERENTIATION: Differences are minimal")
    else:
        print("[OK] DIFFERENTIATION OK: Different inputs -> Different Q-values")

    if scaler is not None:
        scaled_diff = np.max(np.abs(
            results[0]['scaled'][:5] - results[1]['scaled'][:5]
        ))
        print(f"\n[CHECK] Scaler Check:")
        print(f"  Scaled state difference (first 5): {scaled_diff:.6f}")
        if scaled_diff < 0.001:
            print("[FAIL] SCALER ISSUE: Producing similar scaled states")
        else:
            print("[OK] SCALER OK: Producing different scaled states")
    else:
        print("\n(Scaler unavailable - raw features used)")

    print("\n" + "=" * 80)
    print("[VERDICT] FINAL SUMMARY")
    print("=" * 80)

    if diagnosis == "FROZEN":
        print("\n[ALERT] ACTION REQUIRED: Model re-export needed")
        print("   1. Check best checkpoint under data/checkpoints/")
        print("   2. Re-export via scripts/re_export_rl_model.py")
        print("   3. Implement fallback logic immediately")
    elif diagnosis == "LOW_VARIANCE":
        print("\n[WARN] ACTION RECOMMENDED: Implement fallback logic")
        print("   1. Bypass when q_std < 0.001")
        print("   2. Use strategy signal as fallback")
        print("   3. Log bypass events in telemetry")
    else:
        print("\n[OK] NO CRITICAL ACTION NEEDED: Model is healthy")
        print("   Continue monitoring Q-std telemetry")

    print("\n" + "=" * 80)
    print("[OK] Diagnostic Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
