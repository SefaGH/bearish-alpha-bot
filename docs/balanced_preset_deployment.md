# Balanced Preset (500 USDT) Deployment Guide

## Final values (balanced preset)
- risk.equity_usd = 500
- risk.per_trade_risk_pct = 0.003  # 0.3%
- risk.max_position_size = 0.25
- risk.max_notional_pct_per_trade = 0.25
- risk.min_stop_pct = 0.005
- risk.position_size_policy = clip
- risk.size_planner_enabled = true

## Source-of-truth order
1) Azure App Configuration keys (preferred) — label `production` on `appcs-bearish-bot`.
2) Environment variables (only if already present): CAPITAL_USDT, PER_TRADE_RISK_PCT. Do not add new env keys for other risk fields; keep them in App Config.
3) YAML fallback: config/config.example.yaml carries the same balanced defaults.

## Azure App Configuration commands
Use Azure CLI (requires `az login`) and set values for label `production`:

```pwsh
$store = "appcs-bearish-bot"
$label = "production"
$keys = @{
  "risk.equity_usd" = "500"
  "risk.per_trade_risk_pct" = "0.003"
  "risk.max_position_size" = "0.25"
  "risk.max_notional_pct_per_trade" = "0.25"
  "risk.min_stop_pct" = "0.005"
  "risk.position_size_policy" = "clip"
  "risk.size_planner_enabled" = "true"
}

foreach ($k in $keys.Keys) {
  az appconfig kv set --name $store --label $label --key $k --value $keys[$k] --yes
}

az appconfig kv list --name $store --label $label --key "risk.*" --resolve-keyvault
```

Portal steps (alternative): App Configuration ➜ Configuration Explorer ➜ Add/Update each key above with label `production` (no content type needed).

## Verification steps
1) Planner flag: inside the Azure container run `python scripts/print_risk_planner_flag.py` and expect `config_value=true`, `resolved_mode=active`.
2) Paper session (20–30 min): confirm logs show ~500 equity, ~0.3% risk (~$1.50), 25% caps, `[RISK-PLANNER-FLAG] ... resolved_mode=active`, planner decisions with `mode: 'active'`, and `ENQUEUED` sizes matching `planned_notional`.
3) If `resolved_mode` is not `active`: check env overrides (`RISK_SIZE_PLANNER_ENABLED`), ensure App Config keys above exist and match values, then rerun the flag script.
