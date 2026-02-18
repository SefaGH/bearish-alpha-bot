from src.core.production_coordinator import ProductionCoordinator


class _StubStrategyCoordinator:
    def __init__(self):
        self.refresh_calls = []

    def refresh_ml_governance_modes(self, ml_cfg=None):
        governance = {}
        if isinstance(ml_cfg, dict):
            governance = ml_cfg.get("governance", {}) if isinstance(ml_cfg.get("governance"), dict) else {}
        self.refresh_calls.append(governance.get("ppo_mode"))
        return {
            "gemma_mode": governance.get("gemma_mode", "apply"),
            "ppo_mode": governance.get("ppo_mode", "apply"),
        }


def _build_config(*, ppo_mode: str, automation_enabled: bool, auto_recover: bool = False):
    return {
        "monitoring": {"rl_telemetry_interval_seconds": 60},
        "ml": {
            "governance": {
                "gemma_mode": "apply",
                "ppo_mode": ppo_mode,
                "automation": {
                    "enabled": automation_enabled,
                    "interval_sec": 60,
                    "ppo": {
                        "enabled": True,
                        "min_window_samples": 5,
                        "degrade_after_windows": 2,
                        "recover_after_windows": 2,
                        "cooldown_sec": 0,
                        "auto_recover": auto_recover,
                        "degrade_to_mode": "shadow",
                        "bad_flat_vote_rate": 0.95,
                        "bad_avg_score_max": 0.20,
                        "good_flat_vote_rate": 0.40,
                        "good_avg_score_min": 0.60,
                    },
                },
            },
            "reinforcement_learning": {"ppo_enabled": True},
        },
    }


def test_ppo_governance_automation_degrades_after_bad_windows():
    coordinator = ProductionCoordinator(config=_build_config(ppo_mode="apply", automation_enabled=True))
    coordinator.strategy_coordinator = _StubStrategyCoordinator()

    first = coordinator._run_ml_governance_automation_cycle(
        {
            "ppo_samples": 10,
            "ppo_flat_votes": 10,
            "ppo_long_votes": 0,
            "ppo_avg_score": 0.05,
        },
        now_monotonic=100.0,
    )
    assert first["status"] == "tracking"
    assert first["mode"] == "apply"

    second = coordinator._run_ml_governance_automation_cycle(
        {
            "ppo_samples": 20,
            "ppo_flat_votes": 20,
            "ppo_long_votes": 0,
            "ppo_avg_score": 0.04,
        },
        now_monotonic=160.0,
    )
    assert second["status"] == "transition_applied"
    assert second["mode"] == "shadow"
    assert second["transition"]["reason_code"] == "ml.governance.ppo.auto.degrade.flat_vote_rate_high"
    assert coordinator.config["ml"]["governance"]["ppo_mode"] == "shadow"
    assert coordinator.strategy_coordinator.refresh_calls[-1] == "shadow"


def test_ppo_governance_automation_recovers_to_apply_when_enabled():
    coordinator = ProductionCoordinator(
        config=_build_config(ppo_mode="shadow", automation_enabled=True, auto_recover=True)
    )
    coordinator.strategy_coordinator = _StubStrategyCoordinator()
    coordinator._ml_governance_runtime_state["ppo"] = {
        "bad_windows": 0,
        "good_windows": 0,
        "last_transition_mono": 0.0,
        "last_transition_reason_code": "ml.governance.ppo.auto.degrade.flat_vote_rate_high",
        "last_transition_source": "auto",
    }

    first = coordinator._run_ml_governance_automation_cycle(
        {
            "ppo_samples": 10,
            "ppo_flat_votes": 1,
            "ppo_long_votes": 9,
            "ppo_avg_score": 0.80,
        },
        now_monotonic=100.0,
    )
    assert first["status"] == "tracking"
    assert first["mode"] == "shadow"

    second = coordinator._run_ml_governance_automation_cycle(
        {
            "ppo_samples": 20,
            "ppo_flat_votes": 2,
            "ppo_long_votes": 18,
            "ppo_avg_score": 0.82,
        },
        now_monotonic=160.0,
    )
    assert second["status"] == "transition_applied"
    assert second["mode"] == "apply"
    assert second["transition"]["reason_code"] == "ml.governance.ppo.auto.recover.health_window"
    assert coordinator.config["ml"]["governance"]["ppo_mode"] == "apply"
    assert coordinator.strategy_coordinator.refresh_calls[-1] == "apply"


def test_ppo_governance_automation_disabled_keeps_mode():
    coordinator = ProductionCoordinator(config=_build_config(ppo_mode="apply", automation_enabled=False))
    coordinator.strategy_coordinator = _StubStrategyCoordinator()

    snapshot = coordinator._run_ml_governance_automation_cycle(
        {
            "ppo_samples": 10,
            "ppo_flat_votes": 10,
            "ppo_long_votes": 0,
            "ppo_avg_score": 0.01,
        },
        now_monotonic=100.0,
    )

    assert snapshot["status"] == "disabled"
    assert snapshot["mode"] == "apply"
    assert coordinator.config["ml"]["governance"]["ppo_mode"] == "apply"
    assert coordinator.strategy_coordinator.refresh_calls == []
