# Implementation Report: Operational Schema + Log Classification

## Summary
- Added an operational schema for known runtime keys outside canonical YAML.
- Integrated canonical + operational schema into post-merge casting.
- Log classification now separates known runtime keys (info) from truly unknown keys (warning or strict fail).

## Operational Schema (Known Extras)
- bingx_rest_debug: bool (source=runtime) - Enable BingX REST debug logging
- ccxt_timeout_ms: int (source=runtime) - CCXT request timeout in milliseconds
- debug_mode: bool (source=runtime) - Global debug logging toggle
- exchanges: list[str] (source=runtime) - Comma-separated exchange list
- log_level: str (source=runtime) - Logging level (e.g., INFO, DEBUG)
- pythonpath: str (source=runtime) - Python module search path
- pythonunbuffered: int (source=runtime) - Python unbuffered IO flag
- telegram_chat_id: int (source=runtime) - Telegram chat ID
- ticker_cache_ttl_s: float (source=runtime) - Ticker cache TTL in seconds
- ticker_max_attempts: int (source=runtime) - Ticker retry max attempts
- ticker_retry_base_delay_s: float (source=runtime) - Ticker retry base delay in seconds
- trading_duration: int (source=runtime) - Trading duration in seconds
- trading_mode: str (source=runtime) - Trading mode (paper/live)

## Casting Integration
- After merge:
  - Canonical schema paths -> schema-first casting
  - Operational schema paths -> operational casting
  - Schema-unknown strings -> heuristic fallback
- Casting only applies when the runtime value is a string; typed values are left intact.

## Log Classification
- Class A (Known Runtime Keys):
  - Present in operational schema but not canonical
  - Logged at INFO (no warning noise)
- Class B (Truly Unknown Keys):
  - Not in canonical or operational schema
  - Logged at WARNING; with CONFIG_STRICT_TYPE_CHECK=true, fail-fast (ValueError)

## Shadowing / Precedence
- If a key exists in both canonical and operational schema:
  - WARNING (or error in strict mode)
  - Canonical schema wins for casting

## Tests
- `pytest tests/test_live_trading_config.py -k "operational or unknown_appconfig"` -> PASS
- Coverage:
  - Operational key casting (debug_mode, ccxt_timeout_ms, ticker_cache_ttl_s, telegram_chat_id, exchanges)
  - Log classification: known runtime -> non-warning, unknown -> warning
  - Strict mode: unknown -> ValueError
