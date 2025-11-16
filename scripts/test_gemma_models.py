#!/usr/bin/env python3
"""Test GEMMA models after training."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch


def test_model(model_name: str, model_path: str, scaler_path: str | None = None) -> bool:
    """Load a GEMMA model and run a simple inference check."""
    print(f"\n[TEST] {model_name}")

    try:
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"   [ERROR] Model not found: {model_path}")
            return False

        model = torch.jit.load(model_file)
        model.eval()
        print(f"   [OK] Model loaded: {model_path}")

        scaler = None
        if scaler_path:
            scaler_file = Path(scaler_path)
            if scaler_file.exists():
                scaler = joblib.load(scaler_file)
                print(f"   [OK] Scaler loaded: {scaler_path}")
            else:
                print(f"   [WARN] Scaler not found: {scaler_path}")

        dummy_input = np.random.randn(5, 82).astype(np.float32)
        if scaler is not None:
            try:
                dummy_input = scaler.transform(dummy_input)
            except Exception as exc:
                print(f"   [WARN] Scaler transform failed ({exc}); using raw input")

        input_tensor = torch.as_tensor(dummy_input, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)

        print(f"   [OK] Inference succeeded")
        print(f"   Input shape: {tuple(input_tensor.shape)}")
        print(f"   Output shape: {tuple(output.shape)}")

        if output.ndim == 2 and output.shape[1] == 3:
            probs = torch.softmax(output, dim=1)
            classes = ("Bullish", "Neutral", "Bearish")
            for idx in range(min(3, probs.shape[0])):
                pred_class = torch.argmax(probs[idx]).item()
                confidence = probs[idx, pred_class].item()
                print(
                    f"   Sample {idx + 1}: {classes[pred_class]} "
                    f"(confidence={confidence:.2%})"
                )

        return True

    except Exception as exc:  # pylint: disable=broad-except
        print(f"   [ERROR] Test failed: {exc}")
        return False


def main() -> int:
    print("=" * 60)
    print("GEMMA MODEL TESTING")
    print("=" * 60)

    models_to_test = [
        (
            "GEMMA Price",
            "data/models/final/gemma_price.pt",
            "data/models/final/gemma_price_scaler.joblib",
        ),
        (
            "GEMMA Regime",
            "data/models/final/gemma_regime.pt",
            "data/models/final/gemma_regime_scaler.joblib",
        ),
    ]

    results: list[tuple[str, bool]] = []
    for model_name, model_path, scaler_path in models_to_test:
        success = test_model(model_name, model_path, scaler_path)
        results.append((model_name, success))

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for model_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{model_name}: {status}")
        if not success:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All GEMMA models tested successfully.")
        print("Next steps:")
        print("1. Activate GEMMA: ./scripts/activate_gemma.sh")
        print("2. Run paper trading: ./scripts/launch_gemma_test.sh")
    else:
        print("Some tests failed. Review errors above and retrain if required.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
