# GitHub Actions Workflow’ları

## Ortak Kurulum
Tüm job’larda:
```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"

- name: Install deps
  run: |
    python -V
    pip install --upgrade pip setuptools wheel
    pip install --no-cache-dir -r requirements.txt
```

## 1) Run Bot Once
`/.github/workflows/run_bot_once.yml`
- Manuel çalıştırma; Telegram + CSV (paper mode)
- Artefact yok uyarısını önlemek için:
  ```yaml
  - uses: actions/upload-artifact@v4
    with:
      name: bot-run
      path: data/**
      if-no-files-found: ignore
  ```

## 1.5) Run Bot Live
`/.github/workflows/run_bot_live.yml`
- Manuel çalıştırma; GERÇEK emir girişi (live mode)
- **Onay gerekli:** `confirm_live` girişi `YES` olmalı
- Seçilebilir exchange (varsayılan: kucoinfutures)
- Minimal güvenli risk overrides
- Yalnızca `EXECUTION_EXCHANGE` üzerinde emir girer
- Artefact: `data/**`, `state.json`, `day_stats.json`

## 2) Backtest OB
`/.github/workflows/backtest.yml`
- Girdiler: `BT_SYMBOL`, `BT_EXCHANGE`, `BT_LIMIT`
- Çalıştırdığı script: `src/backtest/param_sweep.py`
- Artefact: `backtest-results` (CSV)

## 3) Backtest STR
`/.github/workflows/backtest_str.yml`
- Girdiler: `BT_SYMBOL`, `BT_EXCHANGE`, `BT_LIMIT_30M`, `BT_LIMIT_1H`
- Script: `src/backtest/param_sweep_str.py`
- Artefact: `backtest-results-str` (CSV)

## 4) Nightly (OB+STR + Rapor)
`/.github/workflows/nightly_backtests.yml`
- Cron: `30 23 * * *` (UTC) ≈ 02:30 Türkiye
- VB: bash döngüsüyle virgüllü sembolleri işler
- Özet: `scripts/summarize_backtests.py` → `data/backtests/REPORT.md`
- Artefact: `nightly-backtest-report`
