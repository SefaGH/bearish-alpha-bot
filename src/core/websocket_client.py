"""
ARCHIVED - Generic WebSocket Client

This file is preserved for historical reference but is no longer actively used
for the BingX connection in the production bot. The bot now uses the dedicated
`websocket_client_bingx.py` for all BingX WebSocket communications.

This client was originally designed to handle multiple exchanges via CCXT Pro
and contained fallback logic. This has been replaced by a more direct and
simplified architecture to improve stability and maintainability.

Do not import or use this class for new development.

Last Author: SefaGH
Date Archived: 2025-10-29
"""

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "The generic `websocket_client.py` is loaded but is marked as ARCHIVED. "
    "The system should be using dedicated clients like `websocket_client_bingx.py`."
)

class WebSocketClient:
    """
    ARCHIVED - This class is no longer in active use.
    Using it will raise an error to prevent accidental use in production.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "This generic WebSocketClient is archived and should not be instantiated. "
            "Use a dedicated client (e.g., from websocket_client_bingx.py) instead."
        )

# It is safe to remove all other code from this file.
# This stub ensures that any accidental imports will fail loudly.
