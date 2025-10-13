import os
import logging
from typing import Dict
from .ccxt_client import CcxtClient

logger = logging.getLogger(__name__)

SUPPORTED_EXCHANGES = {
    'binance', 'bingx', 'bitget', 'kucoin', 'kucoinfutures', 'ascendex', 
    'bybit', 'okx', 'gateio', 'mexc'
}

def build_clients_from_env() -> Dict[str, CcxtClient]:
    """
    Build exchange clients from environment variables.
    
    Reads EXCHANGES env var (comma-separated list) and creates authenticated
    clients for each exchange with credentials from {EXCHANGE}_KEY, {EXCHANGE}_SECRET,
    and optionally {EXCHANGE}_PASSWORD environment variables.
    
    Returns:
        Dictionary mapping exchange name to CcxtClient instance
    
    Raises:
        ValueError: If EXCHANGES is empty or contains invalid exchange names
        Warning: Logs warning if exchange credentials are missing
    """
    exchanges_str = os.getenv('EXCHANGES', '').strip()
    if not exchanges_str:
        raise ValueError("EXCHANGES environment variable is empty or not set")
    
    ex_names = [e.strip().lower() for e in exchanges_str.split(',') if e.strip()]
    if not ex_names:
        raise ValueError("EXCHANGES environment variable contains no valid exchange names")
    
    # Validate exchange names
    invalid = [name for name in ex_names if name not in SUPPORTED_EXCHANGES]
    if invalid:
        logger.warning(f"Unknown exchange names: {', '.join(invalid)}. Supported exchanges: {', '.join(sorted(SUPPORTED_EXCHANGES))}")
    
    clients = {}
    for name in ex_names:
        creds = {}
        up = name.upper()
        
        # Special handling: both kucoin and kucoinfutures can use KUCOIN_* credentials
        if name in ('kucoin', 'kucoinfutures'):
            key = os.getenv('KUCOIN_KEY') or os.getenv(f'{up}_KEY')
            sec = os.getenv('KUCOIN_SECRET') or os.getenv(f'{up}_SECRET')
            pwd = os.getenv('KUCOIN_PASSWORD') or os.getenv(f'{up}_PASSWORD')
            logger.debug(f"Loading credentials for {name}: using KUCOIN_* or {up}_* environment variables")
        else:
            key = os.getenv(f'{up}_KEY')
            sec = os.getenv(f'{up}_SECRET')
            pwd = os.getenv(f'{up}_PASSWORD')
            logger.debug(f"Loading credentials for {name}: using {up}_* environment variables")
        
        if not (key and sec):
            logger.warning(f"Missing credentials for {name} (KEY or SECRET not set), skipping")
            continue
        
        logger.debug(f"Credentials found for {name}: KEY={'✓' if key else '✗'}, SECRET={'✓' if sec else '✗'}, PASSWORD={'✓' if pwd else '✗'}")
        
        creds = {'apiKey': key, 'secret': sec}
        if pwd:
            creds['password'] = pwd
        
        try:
            clients[name] = CcxtClient(name, creds)
            logger.info(f"Initialized exchange client: {name}")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {type(e).__name__}: {e}")
            continue
    
    if not clients:
        raise ValueError("No exchange clients could be initialized. Check credentials.")
    
    return clients
