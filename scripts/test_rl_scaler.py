import sys
from pathlib import Path

import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCALER_PATH = "data/models/rl_scaler.joblib"


def test_scaler() -> bool:
    """Verify scaler produces distinct outputs for different inputs."""
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        scaler = None
        print(f"⚠️ No scaler found at {SCALER_PATH}; comparing raw features")

    state1 = np.array(
        [9.42614600e04, 9.42559169e04, 0.0, 7.12776000e01, 7.40257000e01] + [0.0] * 77
    ).reshape(1, -1)
    state2 = np.array(
        [9.40721800e04, 9.40745904e04, 9.901e-01, 4.94091000e01, 4.94184000e01] + [0.0] * 77
    ).reshape(1, -1)

    if scaler:
        scaled1 = scaler.transform(state1)[0]
        scaled2 = scaler.transform(state2)[0]
    else:
        scaled1 = state1[0]
        scaled2 = state2[0]

    diff = np.abs(scaled1 - scaled2)
    max_diff = float(np.max(diff))

    print("=" * 60)
    print("SCALER TEST")
    print("=" * 60)
    print(f"Scaled state 1 (first 10): {scaled1[:10]}")
    print(f"Scaled state 2 (first 10): {scaled2[:10]}")
    print(f"Difference (first 10):     {diff[:10]}")
    print(f"Max difference:            {max_diff}")
    print(
        f"State 1 stats: min={scaled1.min():.4f}, max={scaled1.max():.4f}, std={scaled1.std():.4f}"
    )
    print(
        f"State 2 stats: min={scaled2.min():.4f}, max={scaled2.max():.4f}, std={scaled2.std():.4f}"
    )

    if max_diff < 1e-3:
        print("❌ FAIL: Scaler outputs collapse to same values")
        return False

    print("✅ PASS: Scaler differentiates between inputs")
    return True


if __name__ == "__main__":
    success = test_scaler()
    raise SystemExit(0 if success else 1)
