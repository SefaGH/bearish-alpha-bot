"""
BingX API Authentication Module

Provides HMAC-SHA256 signature generation for BingX private endpoints.
Based on official BingX API documentation.
"""

import hmac
import hashlib
import time
import urllib.parse
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BingXAuthenticator:
    """
    BingX API authentication with HMAC-SHA256 signature generation.
    
    Based on official BingX documentation:
    - X-BX-APIKEY header required
    - timestamp + signature parameters required
    - Signature = HMAC-SHA256(secret, parameter_string)
    """
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Initialize BingX authenticator.
        
        Args:
            api_key: BingX API key
            secret_key: BingX API secret key
        """
        self.api_key = api_key
        self.secret_key = secret_key
        logger.info("🔐 [BINGX-AUTH] Authenticator initialized")
    
    def get_timestamp_ms(self) -> int:
        """
        Get current timestamp in milliseconds.
        
        Returns:
            Current timestamp in milliseconds
        """
        return int(time.time() * 1000)
    
    def generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC-SHA256 signature for BingX API.
        
        Args:
            params: Request parameters dictionary
            
        Returns:
            Hexadecimal signature string
        """
        # Parameter string (NO sorting for query string requests)
        param_string = urllib.parse.urlencode(params)
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.secret_key.encode(), 
            param_string.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"🔐 [BINGX-AUTH] Generated signature")
        return signature
    
    def prepare_authenticated_request(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare authenticated request parameters and headers.
        
        Args:
            params: Optional request parameters
            
        Returns:
            Dictionary with 'params' and 'headers' keys
        """
        if params is None:
            params = {}
        
        # Add required parameters
        params['timestamp'] = self.get_timestamp_ms()
        params['recvWindow'] = 5000
        
        # Generate signature
        signature = self.generate_signature(params)
        params['signature'] = signature
        
        # Prepare headers
        headers = {
            'X-BX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        return {
            'params': params,
            'headers': headers
        }
    
    def convert_symbol_to_bingx(self, ccxt_symbol: str) -> str:
        """
        Convert CCXT symbol format to BingX format.
        
        Args:
            ccxt_symbol: CCXT format symbol (e.g., 'BTC/USDT:USDT')
            
        Returns:
            BingX format symbol (e.g., 'BTC-USDT')
        
        Examples:
            BTC/USDT:USDT → BTC-USDT
            ETH/USDT:USDT → ETH-USDT
            BTC/USDT → BTC-USDT
        """
        # BTC/USDT:USDT → BTC-USDT
        if ":USDT" in ccxt_symbol:
            ccxt_symbol = ccxt_symbol.replace(":USDT", "")
        return ccxt_symbol.replace("/", "-")
