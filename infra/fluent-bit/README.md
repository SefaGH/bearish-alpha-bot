# Fluent Bit (Retired)

This folder contains historical configuration and scripts for a Fluent Bit-based reporting/ingestion pipeline.

## Status

**RETIRED / DEPRECATED for the current production environment.**

The Fluent Bit pipeline was disabled/removed after it contributed to VM instability (retry/log-flood behavior when endpoints/DNS are misconfigured) and because earlier deployment approaches embedded secrets into config files.

## Safety Guard

The scripts below now refuse to run unless you explicitly opt in:

- `deploy_vm.sh`
- `install_fluent_bit.sh`

To bypass the guard (not recommended), set:

- `I_UNDERSTAND_FLUENT_BIT_IS_RETIRED=1`

## Recommendation

Prefer the current Azure reporting automation path (Function/Storage-based flow) instead of reintroducing VM-side Fluent Bit ingestion.
