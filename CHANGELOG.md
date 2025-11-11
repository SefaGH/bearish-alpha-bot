# Değişiklik Günlüğü / Changelog

## [0.1-phase1] - 2025-11-11

### Phase-1 Completion: Infrastructure Setup & Test Hygiene

#### Added
- **CONTRIBUTING.md** - Comprehensive contributor guide with Python 3.11 setup
- **Branch Protection Policy** - Documented in `.github/docs/branch_protection.md`
- **CI Health Report** - Detailed analysis in `diagnostics/ci_health_report.md`
- **Phase-2 Planning** - Kickoff document in `docs/PHASE2_KICKOFF_ISSUE.md`
- **Release Notes** - v0.1-phase1 in `docs/RELEASE_NOTES_v0.1-phase1.md`
- **VERSION file** - Version tracking

#### Changed
- **Python Version** - ⚠️ CRITICAL: Standardized on Python 3.11 (NOT 3.12)
  - Reason: aiohttp 3.8.6 compatibility
  - Updated all documentation
  - Added prominent warnings in README
- **mypy.ini** - Simplified configuration, removed broad `ignore_errors`
- **README.md** - Added Python 3.11 requirement banner
- **src/ml/strategy_integration.py** - Better documented type ignore

#### Fixed
- **Pytest Collection** - Fixed encoding issues and import-time side effects (PR #345)
- **UTF-8 Encoding** - Guaranteed across all text file operations
- **Import Side Effects** - Removed from test modules and diagnostics

#### Security
- Added security guidelines in CONTRIBUTING.md
- Configured bandit security scanning
- Documented secure environment variable usage

---

## 2025-10-12
- **Python 3.12**’ye geçiş; `requirements.txt` 3.12 uyumlu hale getirildi
  - `pandas-ta==0.4.67b0`, `numpy>=2.2.6`, `pandas>=2.2.3,<3`
- `src/core/ccxt_client.py`:
  - `ohlcv()` retry loop düzeltildi; son istisna aynen fırlatılıyor
- `.github/README_BOT.md` ve `README.md` güncellendi (çalışma şekli + kurulum)
- `src/main.py`:
  - veri yeterlilik guard’ları eklendi (min bar kontrolü, `dropna()`)
  - sinyal çağrılarında min 50 bar şartı
  - her koşuda `data/RUN_SUMMARY.txt` yazımı (artifact garantisi)
- **Backtest araçları** eklendi:
  - `src/backtest/param_sweep.py` (OB)
  - `src/backtest/param_sweep_str.py` (STR)
  - Nightly workflow ve raporlama (`scripts/summarize_backtests.py`)
- Workflow düzeltmeleri:
  - Nightly için matrix yerine **bash döngüsü** (split fonksiyonu hatası giderildi)
  - Upload artifact adımlarında `if-no-files-found: ignore` önerisi
- Dokümantasyonda yazım/isim tutarlılığı:
  - `dokümantasyon` yazımı
  - Risk belge anahtarları (cool_down_min/cooldown_min notu giderildi)
