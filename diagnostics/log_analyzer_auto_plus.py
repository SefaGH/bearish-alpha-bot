# log_analyzer_auto_plus.py  →  EN GÜÇLÜ VERSİYON (Mevcut + Gelecek Proof)
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def find_latest_log(): 
    """Geçici: Bugünkü 20 dakikalık paper seans logunu sabit tam dosya yolu ile oku."""
    logs_dir = Path("logs")
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"logs klasörü bulunamadı: {logs_dir.resolve()}")

    candidates = sorted(logs_dir.glob("live_trading_*.log"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"logs/ altında live_trading_*.log bulunamadı: {logs_dir.resolve()}")

    path = candidates[-1]
    print(f"✅ Analiz edilen dosya → {path.name}")
    print(f"   Oluşturma: {datetime.fromtimestamp(path.stat().st_mtime)}\n")
    return path

log_file = find_latest_log()
content = log_file.read_text(encoding="utf-8")

# ================== 1. Detaylı TRADE_CLOSED var mı? ==================
if "TRADE_CLOSED" in content and "trade_id=" in content:
    print("🎉 Issue #417 uygulanmış! Detaylı trade logları kullanılıyor.\n")
    # (önceki scriptteki JSON parse burada devreye girer)
else:
    print("⚠️  Henüz detaylı trade log yok → Toplu özetlerden maksimum analiz yapıyorum...\n")

# ================== 2. YENİ: Session Süresi & Trade Yoğunluğu ==================
start_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*BEARISH ALPHA BOT - STARTING", content)
end_match   = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Bot shutdown complete", content)

duration = None
total_trades = 0

if start_match and end_match:
    start = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S")
    duration = end - start

    total_exits_match = re.search(r"Total Exits:\s*(\d+)", content)
    if total_exits_match:
        total_trades = int(total_exits_match.group(1))
    else:
        total_trades = 0

    print(f"📅 Session süresi      : {duration}")
    if total_trades > 0 and duration.total_seconds() > 0:
        print(f"⚡ Trade/saat          : {total_trades / (duration.total_seconds() / 3600):.2f}")
        print(f"⚡ Trade/dakika        : {total_trades / (duration.total_seconds() / 60):.2f}\n")
    else:
        print("⚡ Trade/saat          : 0.00 (hiç exit yok)")
        print("⚡ Trade/dakika        : 0.00 (hiç exit yok)\n")

# ================== 3. YENİ: Risk Rule Rejects Analizi ==================
rejects = len(re.findall(r"PositionSizeRule.*REJECTED", content)) + \
          len(re.findall(r"REJECTED .*Risk Check", content)) + \
          len(re.findall(r"DailyLossLimitRule", content))
print(f"🚫 Reddedilen sinyal   : {rejects}")
if rejects > 0:
    print("   → MAX_POSITION_SIZE_PCT = 0.2 çok sıkı olabilir!\n")

# ================== 4. YENİ: Regime & ML Kararlılık Analizi ==================
regime_lines = [l for l in content.splitlines() if "Prediction: " in l and "confidence" in l]
confidences = []
if regime_lines:
    for l in regime_lines:
        m = re.search(r"confidence: ([0-9.]+)", l)
        if m:
            confidences.append(float(m.group(1)))

if confidences:
    avg_conf = sum(confidences)/len(confidences)
    low_conf = sum(1 for c in confidences if c < 0.3) / len(confidences) * 100
    print(f"🧠 Regime tahmin sayısı : {len(confidences)}")
    print(f"   Ortalama confidence : {avg_conf:.3f}")
    print(f"   Düşük güven (<0.30) : {low_conf:.1f}% → Hard reject devreye giriyor!\n")
else:
    low_conf = 0

# ================== 5. Gelişmiş Toplu Performans ==================
total_pnl = re.search(r"Total P&L:\s*\S+\s*([+\-]?\d+\.\d+)", content)
wins      = re.search(r"Total Wins:\s*\S+\s*([+\-]?\d+\.\d+)", content)
losses    = re.search(r"Total Losses:\s*\S+\s*([+\-]?\d+\.\d+)", content)
win_rate  = re.search(r"Win Rate:\s*(\d+\.\d+)%", content)
avg_win   = re.search(r"Avg Win:\s*\S+\s*([+\-]?\d+\.\d+)", content)
avg_loss  = re.search(r"Avg Loss:\s*\S+\s*([+\-]?\d+\.\d+)", content)

print("="*70)
print(" GENEL PERFORMANS RAPORU".center(70))
print("="*70)
print(f"Toplam Trade     : {total_trades}")
print(f"Win Rate         : {win_rate.group(1) if win_rate else 'N/A'}%")
print(f"Toplam P&L       : {total_pnl.group(1) if total_pnl else 'N/A'} USDT")
print(f"Kazançlar        : {wins.group(1) if wins else 'N/A'} USDT")
print(f"Kayıplar         : {losses.group(1) if losses else 'N/A'} USDT")
print(f"Ortalama Kazanç  : {avg_win.group(1) if avg_win else 'N/A'} USDT")
print(f"Ortalama Kayıp  : {avg_loss.group(1) if avg_loss else 'N/A'} USDT")

profit_factor = None
if wins and losses and float(losses.group(1)) != 0:
    profit_factor = abs(float(wins.group(1)) / float(losses.group(1)))

expectancy = None
if win_rate and avg_win and avg_loss:
    try:
        win_rate_val = float(win_rate.group(1))
        avg_win_val = abs(float(avg_win.group(1)))
        avg_loss_val = abs(float(avg_loss.group(1)))
        expectancy = (
            (win_rate_val / 100 * avg_win_val)
            - ((100 - win_rate_val) / 100 * avg_loss_val)
        )
    except Exception:
        expectancy = None

print(f"Profit Factor    : {profit_factor:.2f}" if profit_factor is not None else "Profit Factor    : N/A")
print(f"Expectancy       : {expectancy:.3f} USDT/trade" if expectancy is not None else "Expectancy       : N/A")
if total_pnl and duration:
    print(f"Net P&L / Saat   : {(float(total_pnl.group(1)) / (duration.total_seconds() / 3600)):.4f} USDT/saat\n")
else:
    print("Net P&L / Saat   : N/A (eksik veri)\n")

# ================== 6. Öneri Motoru ==================
print("🚀 HEMEN YAPILABİLECEK İYİLEŞTİRMELER")
print("-"*50)
if total_trades > 0 and rejects > total_trades * 0.3:
    print("• MAX_POSITION_SIZE_PCT = 0.2 → 0.3 veya 0.4 yap (çok sinyal reddediliyor)")
if low_conf > 50:
    print("• Regime confidence çok düşük → hard_reject=0.30 → 0.20 düşür")
if profit_factor is not None and profit_factor < 1.3:
    print("• Strateji zarar ettiriyor → RSI threshold'ları gevşet veya regime ignore=True dene")
if duration and total_trades == 0:
    print("• Trade çok az → RSI_RANGE_OB ve RSI_RANGE_STR artır (10 → 15-20)")

# ================== 7. Signal → Trade Funnel ==================
print("\n📥 SIGNAL → TRADE FUNNEL")
print("-"*50)

signals_generated = len(re.findall(r"\[PROCESS\] Processing symbol:", content))
signals_executed = total_trades

print(f"Üretilen sinyal adayları : {signals_generated}")
print(f"Gerçekleşen trade sayısı  : {signals_executed}")

if signals_generated > 0:
    conversion_rate = (signals_executed / signals_generated) * 100
    print(f"Sinyal → Trade dönüşümü : {conversion_rate:.1f}%")
else:
    print("Sinyal → Trade dönüşümü : N/A (sinyal adayı yok)")

print("\nIssue #417 implement edildiğinde bu script 10× daha güçlü olacak!\n")
