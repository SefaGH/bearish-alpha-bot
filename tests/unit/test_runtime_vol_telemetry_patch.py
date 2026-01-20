from __future__ import annotations

from unittest.mock import MagicMock

from src.core.position_manager import AdvancedPositionManager


def test_extract_entry_metadata_maps_vol_telemetry_into_entry_indicators():
    pm = AdvancedPositionManager(risk_manager=MagicMock(), order_manager=MagicMock(), portfolio_manager=MagicMock(cfg={}))

    signal = {
        "timeframe": "1m",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 101.0,
        "entry_indicators": {},
        "meta": {
            "vol_telemetry": {
                "rs_bps": 1.1,
                "gk_bps": 2.2,
                "yz_bps": 3.3,
                "atr_bps": 4.4,
                "std_bps": 5.5,
            }
        },
    }

    out = pm._extract_entry_metadata(signal)
    entry_inds = out.get("entry_indicators")
    entry_lvls = out.get("entry_levels")

    assert isinstance(entry_inds, dict)
    assert entry_inds.get("vol_rs_bps") == 1.1
    assert entry_inds.get("vol_gk_bps") == 2.2
    assert entry_inds.get("vol_yz_bps") == 3.3
    assert entry_inds.get("vol_atr_bps") == 4.4
    assert entry_inds.get("vol_std_bps") == 5.5

    assert isinstance(entry_lvls, dict)
    assert entry_lvls.get("entry_price") == 100.0
    assert entry_lvls.get("stop_price") == 99.0
    assert entry_lvls.get("target_price") == 101.0


def test_extract_entry_metadata_handles_missing_meta_gracefully():
    pm = AdvancedPositionManager(risk_manager=MagicMock(), order_manager=MagicMock(), portfolio_manager=MagicMock(cfg={}))

    signal = {"timeframe": "1m", "entry_indicators": {"rsi": 42.0}}
    out = pm._extract_entry_metadata(signal)

    assert out.get("entry_indicators", {}).get("rsi") == 42.0
