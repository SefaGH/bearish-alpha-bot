"""
Lightweight protocol definitions for core components.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Union


class PositionSizingProtocol(Protocol):
    async def calculate_optimal_size(
        self,
        signal: Dict[str, Any],
        method: str = "fixed_risk_capped",
        return_signal: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], float]:
        ...
