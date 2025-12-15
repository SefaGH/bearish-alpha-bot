"""
Canonical signal intent constants used across the trading pipeline.
"""

INTENT_ENTRY = "entry"
INTENT_SCALE_IN = "scale_in"
INTENT_REENTRY = "reentry"

INTENT_CLOSE = "close"
INTENT_REDUCE = "reduce"
INTENT_REVERSE = "reverse"
INTENT_FORCE_SWAP = "force_swap"

MAINTENANCE_INTENTS = {
    INTENT_CLOSE,
    INTENT_REDUCE,
    INTENT_REVERSE,
    INTENT_FORCE_SWAP,
}
