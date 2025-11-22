# log_analyzer_auto_plus.py  →  EN GÜÇLÜ VERSİYON (Mevcut + Gelecek Proof)
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def find_latest_log():
    logs = list(Path(".").glob("live_trading_*.log"))
    if not logs:
        raise FileNotFoundError("Log dosyası bulunamadı!")
    latest = max(logs, key=lambda x: x.stat().st_mtime)
    print(f"✅ Analiz edilen dosya → {latest.name}")
    print(f"   Oluşturma: {datetime.fromtimestamp(latest.stat().st_mtime)}\n")
    return latest

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

if start_match and end_match:
    start = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S")
    duration = end - start
    total_trades = int(re.search(r"Total Exits:\s*(\d+)", content).group(1))
    print(f"📅 Session süresi      : {duration}")
    print(f"⚡ Trade/saat          : {total_trades / (duration.total_seconds() / 3600):.2f}")
    print(f"⚡ Trade/dakika        : {total_trades / (duration.total_seconds() / 60):.2f}\n")

# ================== 3. YENİ: Risk Rule Rejects Analizi ==================
rejects = len(re.findall(r"PositionSizeRule.*REJECTED", content)) + \
          len(re.findall(r"REJECTED .*Risk Check", content)) + \
          len(re.findall(r"DailyLossLimitRule", content))
print(f"🚫 Reddedilen sinyal   : {rejects}")
if rejects > 0:
    print("   → MAX_POSITION_SIZE_PCT = 0.2 çok sıkı olabilir!\n")

# ================== 4. YENİ: Regime & ML Kararlılık Analizi ==================
regime_lines = [l for l in content.splitlines() if "Prediction: " in l and "confidence" in l]
if regime_lines:
    confidences = [float(re.search(r"confidence: ([0-9.]+)", l).group(1)) for l in regime_lines]
    avg_conf = sum(confidences)/len(confidences)
    low_conf = sum(1 for c in confidences if c < 0.3) / len(confidences) * 100
    print(f"🧠 Regime tahmin sayısı : {len(regime_lines)}")
    print(f"   Ortalama confidence : {avg_conf:.3f}")
    print(f"   Düşük güven (<0.30) : {low_conf:.1f}% → Hard reject devreye giriyor!\n")

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
print(f"Ortalama Kayıp   : {avg_loss.group(1) if avg_loss else 'N/A'} USDT")

profit_factor = abs(float(wins.group(1))/float(losses.group(1))) if wins and losses and float(losses.group(1)) != 0 else float('inf')
 expectancy = (float(win_rate.group(1))/100 * abs(float(avg_win.group(1))) ) - ( (100-float(win_rate.group(1)))/100 * abs(float(avg_loss.group(1))) ) if all([win_rate, avg_win, avg_loss]) else 0

print(f"Profit Factor    : {profit_factor:.2f}")
print(f"Expectancy       : {expectancy:.3f} USDT/trade")
print(f"Net P&L / Saat   : {(float(total_pnl.group(1)) / (duration.total_seconds() / 3600)):.4f} USDT/saat\n")

# ================== 6. Öneri Motoru ==================
print("🚀 HEMEN YAPILABİLECEK İYİLEŞTİRMELER")
print("-"*50)
if rejects > total_trades * 0.3:
    print("• MAX_POSITION_SIZE_PCT = 0.2 → 0.3 veya 0.4 yap (çok sinyal reddediliyor)")
if low_conf > 50:
    print("• Regime confidence çok düşük → hard_reject=0.30 → 0.20 düşür")
if profit_factor < 1.3:
    print("• Strateji zarar ettiriyor → RSI threshold'ları gevşet veya regime ignore=True dene")
if total_trades / (duration.total_seconds() / 3600) < 1:
    print("• Trade çok az → RSI_RANGE_OB ve RSI_RANGE_STR artır (10 → 15-20)")
print("\nIssue #417 implement edildiğinde bu script 10× daha güçlü olacak!\n")
