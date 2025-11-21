import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# --- PATH FIX ---
# Proje kök dizinini (repo root) sys.path'e ekle
# Bu dosya: tests/unit/test_head_scale_migration.py
# Root: ../../
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Artık src ve scripts modüllerine erişebiliriz
try:
    from src.ml.reinforcement_learning import TradingRLAgent, DQNNetwork
except ImportError:
    # Fallback (Eğer src root olarak ayarlıysa)
    from ml.reinforcement_learning import TradingRLAgent, DQNNetwork

def test_legacy_head_scale_migration():
    """
    Tests if a legacy checkpoint (with head_scale_alpha/log) is correctly
    migrated to the new head_scale_raw parameter format upon loading.
    """
    # 1. Setup a dummy agent
    state_size = 10
    action_size = 3
    agent = TradingRLAgent(
        state_size=state_size,
        action_size=action_size,
        config={
            "training_mode": True,
            "head_scale_learnable": True,
            "initial_head_scale": 1.0,
            "head_scale_min_multiplier": 0.1
        }
    )

    # 2. Create a fake legacy checkpoint dictionary
    # Simulating 'head_scale_alpha' which was used in previous versions
    # head_scale = 1.0 + alpha => let's target scale=1.5 => alpha=0.5
    target_scale = 1.5
    min_mult = agent.head_scale_min_multiplier # 0.1
    
    # Legacy alpha logic: scale = clamp(1.0 + alpha, min=min_mult)
    # So alpha = 0.5 should result in scale 1.5
    legacy_alpha = torch.tensor([0.5])
    
    checkpoint = {
        'q_network': {
            'network.0.weight': torch.randn(64, state_size), # Dummy weights to pass load checks
            'head_scale_alpha': legacy_alpha 
            # Note: 'head_scale_raw' is MISSING, simulating old checkpoint
        },
        'target_network': {
             'head_scale_alpha': legacy_alpha
        },
        'optimizer': {}, # Empty optimizer state
        'epsilon': 0.5,
        'training_history': {}
    }
    
    # Save this fake checkpoint to a temporary file
    temp_path = "temp_legacy_ckpt.pth"
    torch.save(checkpoint, temp_path)
    
    try:
        # 3. Load the legacy checkpoint
        # The agent should detect 'head_scale_alpha' and migrate it to 'head_scale_raw'
        agent.load_model(temp_path)
        
        # 4. Verify Migration
        # The q_network should now have 'head_scale_raw' parameter populated
        assert hasattr(agent.q_network, 'head_scale_raw'), "Migration failed: head_scale_raw not created"
        
        # Calculate expected raw value:
        # scale = min_mult + softplus(raw)
        # 1.5 = 0.1 + softplus(raw) => 1.4 = softplus(raw)
        # raw = inverse_softplus(1.4) = log(exp(1.4) - 1)
        expected_raw = float(torch.log(torch.expm1(torch.tensor(target_scale - min_mult))))
        
        actual_raw = float(agent.q_network.head_scale_raw.item())
        
        # Check if values are close enough
        assert abs(actual_raw - expected_raw) < 1e-4, \
            f"Migration value mismatch. Expected raw ~{expected_raw}, got {actual_raw}"
            
        # Verify the final effective scale property
        effective_scale = float(agent.q_network.head_scale.item())
        assert abs(effective_scale - target_scale) < 1e-4, \
            f"Effective scale mismatch. Expected {target_scale}, got {effective_scale}"
            
        print("✅ Legacy migration test passed!")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_legacy_head_scale_migration()
