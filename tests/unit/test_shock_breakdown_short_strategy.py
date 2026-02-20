import pandas as pd

from src.strategies.shock_breakdown_short import ShockBreakdownShortStrategy


def _build_df(
    *,
    strong_momentum: bool = True,
    strong_volume: bool = True,
    breakdown: bool = True,
) -> pd.DataFrame:
    idx = pd.date_range("2026-02-19 14:00:00+00:00", periods=40, freq="5min")
    close = [100.0] * 35
    if strong_momentum:
        tail = [100.0, 99.8, 99.5, 99.2, 98.8]
    else:
        tail = [100.0, 99.98, 99.97, 99.96, 99.95]
    close.extend(tail)

    if not breakdown:
        close[-1] = 100.05

    volume = [100.0] * 39
    volume.append(220.0 if strong_volume else 95.0)

    data = {
        "open": [c + 0.05 for c in close],
        "high": [c + 0.10 for c in close],
        "low": [c - 0.10 for c in close],
        "close": close,
        "volume": volume,
    }
    return pd.DataFrame(data, index=idx)


def _base_cfg() -> dict:
    return {
        "enabled": True,
        "timeframe": "5m",
        "shock_state": "ARMED",
        "min_shock_score": 0.60,
        "breakdown_lookback_bars": 20,
        "breakdown_confirm_bps": 5.0,
        "momentum_lookback_bars": 3,
        "min_momentum_pct": 0.003,
        "volume_ma_window": 20,
        "min_volume_mult": 1.20,
        "cooldown_seconds": 600,
        "take_profit_pct": 0.010,
        "stop_loss_pct": 0.006,
        "rollout": {"mode": "enforce", "canary_symbols": ["*"]},
        "exit_settings": {"max_hold_seconds": 900},
    }


def test_shock_breakdown_short_emits_signal_when_all_gates_pass():
    strategy = ShockBreakdownShortStrategy(_base_cfg())
    df = _build_df(strong_momentum=True, strong_volume=True, breakdown=True)

    signal = strategy.signal(
        df_30m=None,
        symbol="BTC/USDT:USDT",
        market_data={"5m": df, "shock": {"state": "ARMED", "shock_score": 0.82}},
    )

    assert isinstance(signal, dict)
    assert signal.get("strategy_name") == "shock_breakdown_short"
    assert signal.get("side") == "sell"
    assert signal.get("reason_code") == "strategy.shock_breakdown_short.entry"
    assert signal.get("tp_pct") == 0.01
    assert signal.get("sl_pct") == 0.006
    assert signal.get("meta", {}).get("shock_state") == "ARMED"


def test_shock_breakdown_short_observe_mode_does_not_emit_signal():
    cfg = _base_cfg()
    cfg["rollout"] = {"mode": "observe", "canary_symbols": ["*"]}
    strategy = ShockBreakdownShortStrategy(cfg)
    df = _build_df(strong_momentum=True, strong_volume=True, breakdown=True)

    signal = strategy.signal(
        df_30m=None,
        symbol="BTC/USDT:USDT",
        market_data={"5m": df, "shock": {"state": "ARMED", "shock_score": 0.82}},
    )

    assert signal is None


def test_shock_breakdown_short_canary_filter_blocks_non_canary_symbol():
    cfg = _base_cfg()
    cfg["rollout"] = {"mode": "enforce", "canary_symbols": ["ETH/USDT:USDT"]}
    strategy = ShockBreakdownShortStrategy(cfg)
    df = _build_df(strong_momentum=True, strong_volume=True, breakdown=True)

    signal = strategy.signal(
        df_30m=None,
        symbol="BTC/USDT:USDT",
        market_data={"5m": df, "shock": {"state": "ARMED", "shock_score": 0.90}},
    )

    assert signal is None


def test_shock_breakdown_short_blocks_when_volume_gate_fails():
    strategy = ShockBreakdownShortStrategy(_base_cfg())
    df = _build_df(strong_momentum=True, strong_volume=False, breakdown=True)

    signal = strategy.signal(
        df_30m=None,
        symbol="BTC/USDT:USDT",
        market_data={"5m": df, "shock": {"state": "ARMED", "shock_score": 0.82}},
    )

    assert signal is None


def test_shock_breakdown_short_applies_symbol_cooldown():
    strategy = ShockBreakdownShortStrategy(_base_cfg())
    df = _build_df(strong_momentum=True, strong_volume=True, breakdown=True)
    market_data = {"5m": df, "shock": {"state": "ARMED", "shock_score": 0.82}}

    first = strategy.signal(df_30m=None, symbol="BTC/USDT:USDT", market_data=market_data)
    second = strategy.signal(df_30m=None, symbol="BTC/USDT:USDT", market_data=market_data)

    assert isinstance(first, dict)
    assert second is None
