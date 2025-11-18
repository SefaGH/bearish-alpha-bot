name: 🧠 Train & Validate RL Agent

on:
  workflow_dispatch:
    inputs:
      log_level:
        description: "Python log level"
        required: true
        default: "INFO"
        type: choice
        options:
          - DEBUG
          - INFO
          - WARNING
          - ERROR
      exchange:
        description: "Exchange id (ccxt)"
        required: true
        default: "bingx"
        type: string
      symbol:
        description: "Trading symbol (CCXT format)"
        required: true
        default: "BTC/USDT:USDT"
        type: string
      timeframe:
        description: "Candle timeframe"
        required: true
        default: "1h"
        type: string
      candles:
        description: "Number of candles to fetch for RL dataset"
        required: true
        default: "6000"
        type: string
      episodes:
        description: "RL training episodes (override config ml.reinforcement_learning.training.episodes)"
        required: false
        default: "250"
        type: string
      save_every:
        description: "Checkpoint frequency in episodes"
        required: false
        default: "25"
        type: string
      run_headscale_diagnostics:
        description: "Run head-scale inspectors & migration pytest (Section 4)"
        required: true
        default: "true"
        type: choice
        options:
          - "true"
          - "false"

jobs:
  train-and-validate-rl:
    runs-on: ubuntu-latest

    env:
      PYTHONUNBUFFERED: "1"
      RL_CONFIG_PATH: config/config.example.yaml
      RL_OUTPUT_DIR: data/training
      RL_MODEL_DIR: data/checkpoints

    steps:
      - name: 1. Checkout repository
        uses: actions/checkout@v4

        # GEMMA manifest/bundle hazırlığı
      - name: 2. Setup Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: 3. Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt

      - name: 4. Ensure GEMMA bundle & manifest exist
        run: |
          echo "======================================================================"
          echo "🧩 STEP 4: Ensuring GEMMA bundle and manifest for RL (artifacts/gemma/final)"
          echo "======================================================================"

          BUNDLE_DIR="artifacts/gemma/final"
          MANIFEST_PATH="${BUNDLE_DIR}/manifest.json"

          mkdir -p "${BUNDLE_DIR}"

          if [ -f "${MANIFEST_PATH}" ]; then
            echo "✅ Existing GEMMA manifest found at ${MANIFEST_PATH}"
          else
            echo "⚠️  GEMMA manifest not found at ${MANIFEST_PATH}"
            echo "   Attempting to generate GEMMA bundle via scripts/setup_gemma_artifacts.sh"
            if [ -x "scripts/setup_gemma_artifacts.sh" ]; then
              bash scripts/setup_gemma_artifacts.sh
            else
              echo "❌ scripts/setup_gemma_artifacts.sh is missing or not executable."
              echo "   RL training requires a valid GEMMA manifest.json under artifacts/gemma/final."
              exit 1
            fi

            if [ -f "${MANIFEST_PATH}" ]; then
              echo "✅ GEMMA manifest successfully generated at ${MANIFEST_PATH}"
            else
              echo "❌ Failed to generate GEMMA manifest. Aborting RL training workflow."
              echo "   Please run train-models.yml (Gemma training) and setup_gemma_artifacts.sh first."
              exit 1
            fi
          fi

          echo ""
          echo "ℹ️  RL training will use GEMMA manifest-driven feature extraction (82 features)."

      # 5) RL config sanity: training_mode MUST be true
      - name: 5. Sanity-check RL config (training_mode must be true)
        run: |
          echo "======================================================================"
          echo "🧠 STEP 5: Checking RL config.training_mode for training workflow"
          echo "======================================================================"
          python - << 'PY'
          import os
          from src.config.live_trading_config import LiveTradingConfiguration

          cfg_path = os.environ.get("RL_CONFIG_PATH", "config/config.example.yaml")
          cfg = LiveTradingConfiguration.load(config_path=cfg_path, log_summary=False)
          rl_cfg = (cfg.get("ml") or {}).get("reinforcement_learning", {})
          training_mode = rl_cfg.get("training_mode", False)
          if not training_mode:
              raise SystemExit(
                  "❌ ml.reinforcement_learning.training_mode is FALSE in config.\n"
                  "   RL TRAINING requires training_mode=true. "
                  "Set ML_RL_TRAINING_MODE=true as a GitHub Variable / env override or update config."
              )
          print(f"✅ RL training_mode is TRUE in {cfg_path}. Proceeding with RL training.")
          PY

      # 6) Prepare RL datasets
      - name: 6. Prepare RL training/val/test datasets
        id: prepare_rl
        run: |
          echo "======================================================================"
          echo "📊 STEP 6: Preparing RL dataset with engineered features"
          echo "======================================================================"

          python scripts/prepare_rl_training_data.py \
            --exchange "${{ inputs.exchange }}" \
            --symbol "${{ inputs.symbol }}" \
            --timeframe "${{ inputs.timeframe }}" \
            --candles ${{ inputs.candles }} \
            --output-dir "${RL_OUTPUT_DIR}" \
            --config "${RL_CONFIG_PATH}" \
            --train-ratio 0.7 \
            --val-ratio 0.15 \
            --log-level "${{ inputs.log_level }}"

          python - << 'PY'
          import os
          from scripts.prepare_rl_training_data import sanitize_symbol
          symbol = os.environ.get("SYMBOL", "BTC/USDT:USDT")
          timeframe = os.environ.get("TIMEFRAME", "1h")
          base = f"{sanitize_symbol(symbol)}_{timeframe}"
          print(f"BASE_NAME={base}")
          with open(os.environ["GITHUB_OUTPUT"], "a") as f:
              f.write(f"base_name={base}\n")
          PY
        env:
          SYMBOL: ${{ inputs.symbol }}
          TIMEFRAME: ${{ inputs.timeframe }}

      - name: 7. Train RL agent on train split
        id: train_rl
        run: |
          echo "======================================================================"
          echo "🚀 STEP 7: Training RL Agent"
          echo "======================================================================"

          BASE_NAME="${{ steps.prepare_rl.outputs.base_name }}"
          TRAIN_DATASET="${RL_OUTPUT_DIR}/${BASE_NAME}_train.npz"

          EPISODES=${{ inputs.episodes }}
          SAVE_EVERY=${{ inputs.save_every }}

          echo "Using dataset: ${TRAIN_DATASET}"
          echo "Episodes: ${EPISODES}, Save every: ${SAVE_EVERY}"

          python scripts/train_rl_agent.py \
            --dataset "${TRAIN_DATASET}" \
            --config "${RL_CONFIG_PATH}" \
            --episodes "${EPISODES}" \
            --save-every "${SAVE_EVERY}" \
            --model-dir "${RL_MODEL_DIR}" \
            --model-name "rl_agent_train.pth" \
            --summary-file "${RL_OUTPUT_DIR}/rl_training_summary.json" \
            --log-level "${{ inputs.log_level }}"

          echo "final_checkpoint=${RL_MODEL_DIR}/rl_agent_final.pth" >> "$GITHUB_OUTPUT"

      - name: 8. Validate RL checkpoint on validation split
        id: validate_rl
        run: |
          echo "======================================================================"
          echo "🔬 STEP 8: Validating RL checkpoint on validation split"
          echo "======================================================================"

          BASE_NAME="${{ steps.prepare_rl.outputs.base_name }}"
          VAL_DATASET="${RL_OUTPUT_DIR}/${BASE_NAME}_val.npz"
          CHECKPOINT="${{ steps.train_rl.outputs.final_checkpoint }}"
          REPORT_PATH="${RL_OUTPUT_DIR}/rl_validation_report.json"

          echo "Validation dataset: ${VAL_DATASET}"
          echo "Checkpoint: ${CHECKPOINT}"

          python scripts/validate_rl_model.py \
            --dataset "${VAL_DATASET}" \
            --checkpoint "${CHECKPOINT}" \
            --config "${RL_CONFIG_PATH}" \
            --report-file "${REPORT_PATH}" \
            --log-level "${{ inputs.log_level }}"

          echo "validation_report=${REPORT_PATH}" >> "$GITHUB_OUTPUT"

      - name: 9. Inspect replay buffer & TD normalization (optional)
        if: always()
        run: |
          echo "======================================================================"
          echo "🧩 STEP 9: Optional Replay / TD inspection (if script present)"
          echo "======================================================================"

          if [ -f "scripts/inspect_replay_and_td.py" ]; then
            BASE_NAME="${{ steps.prepare_rl.outputs.base_name }}"
            TRAIN_DATASET="${RL_OUTPUT_DIR}/${BASE_NAME}_train.npz"
            CHECKPOINT="${{ steps.train_rl.outputs.final_checkpoint }}"

            python scripts/inspect_replay_and_td.py \
              --dataset "${TRAIN_DATASET}" \
              --checkpoint "${CHECKPOINT}" \
              --reward-clip 10 \
              --reward-scale 0.01 \
              --log-level "${{ inputs.log_level }}"
          else
            echo "scripts/inspect_replay_and_td.py not found; skipping this diagnostic step."
          fi

      - name: 10. Run head-scale inspectors and migration test
        if: inputs.run_headscale_diagnostics == 'true'
        run: |
          echo "======================================================================"
          echo "🧪 STEP 10: Head-scale diagnostics & migration test"
          echo "======================================================================"

          export PYTHONPATH="src:."
          MODEL_PATH="${{ steps.train_rl.outputs.final_checkpoint }}"

          python scripts/inspect_head_param.py \
            --model "${MODEL_PATH}" \
            --state-size 82 \
            --head-scale-learnable \
            --initial-head-scale 1.0 \
            --head-scale-min 0.1

          python scripts/inspect_head_gradient.py \
            --model "${MODEL_PATH}" \
            --state-size 82 \
            --head-scale-learnable \
            --initial-head-scale 1.0 \
            --head-scale-min 0.1 \
            --batch-size 32

          pytest tests/unit/test_head_scale_migration.py -q

      - name: 11. Summarize RL validation metrics
        run: |
          echo "======================================================================"
          echo "📋 STEP 11: RL Validation Summary"
          echo "======================================================================"
          REPORT="${{ steps.validate_rl.outputs.validation_report }}"
          if [ -f "${REPORT}" ]; then
            python - << 'PY'
            import json, os
            report_path = os.environ["REPORT"]
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("\n--- RL VALIDATION REPORT ---")
            total_reward = data.get("total_reward") or data.get("reward_total") or "N/A"
            pnl_total = data.get("pnl_total", "N/A")
            action_dist = data.get("action_distribution", {})
            q_stats = data.get("q_stats", {})
            print(f"Total Reward: {total_reward}")
            print(f"Total PnL:    {pnl_total}")
            print(f"Action Distribution: {action_dist}")
            print(f"Q-Value Stats: {q_stats}")
            print("--- END REPORT ---\n")
            PY
          else:
            echo "Validation report not found at ${REPORT}"
        env:
          REPORT: ${{ steps.validate_rl.outputs.validation_report }}

      - name: 12. Collect RL training artifacts
        run: |
          echo "======================================================================"
          echo "📦 STEP 12: Collecting RL artifacts for upload"
          echo "======================================================================"

          mkdir -p artifacts/rl

          cp -v "${RL_OUTPUT_DIR}/"*_metadata.json artifacts/rl/ 2>/dev/null || true
          cp -v "${RL_OUTPUT_DIR}/rl_training_summary.json" artifacts/rl/ 2>/dev/null || true
          cp -v "${RL_OUTPUT_DIR}/rl_validation_report.json" artifacts/rl/ 2>/dev/null || true

          cp -v "${RL_MODEL_DIR}/rl_agent_final.pth" artifacts/rl/ 2>/dev/null || true

          if [ -d "${RL_MODEL_DIR}" ]; then
            mkdir -p artifacts/rl/checkpoints
            cp -v "${RL_MODEL_DIR}/"*.pth artifacts/rl/checkpoints/ 2>/dev/null || true
          fi

          if [ -d "logs" ]; then
            mkdir -p artifacts/rl/logs
            cp -rv logs artifacts/rl/logs/ 2>/dev/null || true
          fi

      - name: 13. Upload RL artifacts
        uses: actions/upload-artifact@v4
        with:
          name: rl-agent-training-${{ github.run_number }}
          path: artifacts/rl
          retention-days: 30
