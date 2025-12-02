import os
import azure.functions as func
import logging
import re
from datetime import datetime
from urllib.parse import urlparse

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

STORAGE_CONNECTION = os.environ.get("bearishstorage_STORAGE")
REPORTS_CONTAINER = os.environ.get("REPORTS_CONTAINER", "reports")

# 1. Event Grid trigger listens for blob-created events.
# 2. Blob input binding downloads the payload referenced by the event.
@app.event_grid_trigger(arg_name="event")
@app.blob_input(
    arg_name="inputblob",
    path="{data.url}",  # use dot-notation so binding parses Event Grid payload
    connection="bearishstorage_STORAGE",
)
def ProcessLogFileOnUpload(event: func.EventGridEvent, inputblob: bytes):
    logging.info("Python Event Grid trigger fonksiyonu çalıştı.")
    logging.info(f"Olay Tipi: {event.event_type}")
    event_payload = event.get_json()
    blob_url = event_payload.get("url", "")
    logging.info(f"İşlenen Dosya URL: {blob_url}")

    try:
        if isinstance(inputblob, bytes):
            content = inputblob.decode("utf-8")
        else:
            content = str(inputblob)
        logging.info(f"Dosya boyutu: {len(content)} bytes")
    except Exception as exc:  # noqa: BLE001
        logging.error(f"Dosya içeriği okunamadı veya çözülemedi: {exc}")
        return

    report = analyze_log_content(content)

    logging.info(  
        "\n\n" + "=" * 80 + "\n" + " OTOMATİK LOG ANALİZ RAPORU ".center(80) + "\n" + "=" * 80
    )
    logging.info(report)
    logging.info("=" * 80 + "\n\n")

    report_url = _upload_report(report, blob_url)
    if report_url:
        logging.info(f"Rapor kaydedildi: {report_url}")


def _upload_report(report: str, blob_url: str) -> str:
    if not STORAGE_CONNECTION or not blob_url:
        return ""

    try:
        service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION)
        container = service.get_container_client(REPORTS_CONTAINER)
        try:
            container.create_container()
        except ResourceExistsError:
            pass

        blob_name = _derive_report_blob_name(blob_url)
        container.upload_blob(blob_name, report.encode("utf-8"), overwrite=True)
        return (
            f"https://{service.account_name}.blob.core.windows.net/"
            f"{REPORTS_CONTAINER}/{blob_name}"
        )
    except Exception as exc:  # noqa: BLE001
        logging.error(f"Rapor blob kaydı başarısız: {exc}")
        return ""


def _derive_report_blob_name(blob_url: str) -> str:
    path = urlparse(blob_url).path.strip("/")
    filename = path.rsplit("/", 1)[-1] if path else ""
    if not filename:
        filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f"{base}.report.txt"


def analyze_log_content(content: str) -> str:
    """Verilen log içeriğini (string) analiz eder ve okunabilir bir rapor string'i döndürür."""
    report_lines = []

    if "TRADE_CLOSED" in content and "trade_id=" in content:
        report_lines.append("🎉 Issue #417 uygulanmış! Detaylı trade logları kullanılıyor.\n")
    else:
        report_lines.append(
            "⚠️  Henüz detaylı trade log yok → Toplu özetlerden maksimum analiz yapılıyor...\n"
        )

    start_match = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*BEARISH ALPHA BOT - STARTING",
        content,
    )
    end_match = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Bot shutdown complete", content
    )

    duration = None
    total_trades = 0

    if start_match and end_match:
        start = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S")
        duration = end - start

        total_exits_match = re.search(r"Total Exits:\s*(\d+)", content)
        total_trades = int(total_exits_match.group(1)) if total_exits_match else 0

        report_lines.append(f"📅 Session süresi      : {duration}")
        if total_trades > 0 and duration.total_seconds() > 0:
            report_lines.append(
                f"⚡ Trade/saat          : {total_trades / (duration.total_seconds() / 3600):.2f}"
            )
            report_lines.append(
                f"⚡ Trade/dakika        : {total_trades / (duration.total_seconds() / 60):.2f}\n"
            )
        else:
            report_lines.append("⚡ Trade/saat          : 0.00 (hiç exit yok)")
            report_lines.append("⚡ Trade/dakika        : 0.00 (hiç exit yok)\n")

    rejects = len(re.findall(r"PositionSizeRule.*REJECTED", content)) + len(
        re.findall(r"REJECTED .*Risk Check", content)
    ) + len(re.findall(r"DailyLossLimitRule", content))
    report_lines.append(f"🚫 Reddedilen sinyal   : {rejects}")
    if rejects > 0:
        report_lines.append("   → MAX_POSITION_SIZE_PCT = 0.2 çok sıkı olabilir!\n")

    regime_lines = [l for l in content.splitlines() if "Prediction: " in l and "confidence" in l]
    confidences = [
        float(m.group(1)) for l in regime_lines if (m := re.search(r"confidence: ([0-9.]+)", l))
    ]

    low_conf_rate = 0
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        low_conf_rate = sum(1 for c in confidences if c < 0.3) / len(confidences) * 100
        report_lines.append(f"🧠 Regime tahmin sayısı : {len(confidences)}")
        report_lines.append(f"   Ortalama confidence : {avg_conf:.3f}")
        report_lines.append(
            f"   Düşük güven (<0.30) : {low_conf_rate:.1f}% → Hard reject devreye giriyor!\n"
        )

    total_pnl = re.search(r"Total P&L:\s*\S+\s*([+\-]?\d+\.\d+)", content)
    wins = re.search(r"Total Wins:\s*\S+\s*([+\-]?\d+\.\d+)", content)
    losses = re.search(r"Total Losses:\s*\S+\s*([+\-]?\d+\.\d+)", content)
    win_rate = re.search(r"Win Rate:\s*(\d+\.\d+)%", content)
    avg_win = re.search(r"Avg Win:\s*\S+\s*([+\-]?\d+\.\d+)", content)
    avg_loss = re.search(r"Avg Loss:\s*\S+\s*([+\-]?\d+\.\d+)", content)

    report_lines.append("=" * 70)
    report_lines.append(" GENEL PERFORMANS RAPORU".center(70))
    report_lines.append("=" * 70)
    report_lines.append(f"Toplam Trade     : {total_trades}")
    report_lines.append(f"Win Rate         : {win_rate.group(1) if win_rate else 'N/A'}%")
    report_lines.append(f"Toplam P&L       : {total_pnl.group(1) if total_pnl else 'N/A'} USDT")
    report_lines.append(f"Kazançlar        : {wins.group(1) if wins else 'N/A'} USDT")
    report_lines.append(f"Kayıplar         : {losses.group(1) if losses else 'N/A'} USDT")
    report_lines.append(f"Ortalama Kazanç  : {avg_win.group(1) if avg_win else 'N/A'} USDT")
    report_lines.append(f"Ortalama Kayıp   : {avg_loss.group(1) if avg_loss else 'N/A'} USDT")

    profit_factor = None
    if wins and losses:
        try:
            losses_val = float(losses.group(1))
            if losses_val != 0:
                profit_factor = abs(float(wins.group(1)) / losses_val)
        except (ValueError, IndexError):
            profit_factor = None

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
        except (ValueError, IndexError):
            expectancy = None

    report_lines.append(
        f"Profit Factor    : {profit_factor:.2f}"
        if profit_factor is not None
        else "Profit Factor    : N/A"
    )
    report_lines.append(
        f"Expectancy       : {expectancy:.3f} USDT/trade"
        if expectancy is not None
        else "Expectancy       : N/A"
    )
    if total_pnl and duration and duration.total_seconds() > 0:
        report_lines.append(
            f"Net P&L / Saat   : {(float(total_pnl.group(1)) / (duration.total_seconds() / 3600)):.4f} USDT/saat\n"
        )
    else:
        report_lines.append("Net P&L / Saat   : N/A (eksik veri)\n")

    report_lines.append("🚀 HEMEN YAPILABİLECEK İYİLEŞTİRMELER")
    report_lines.append("-" * 50)
    suggestions = 0
    if total_trades > 0 and rejects > total_trades * 0.3:
        report_lines.append("• MAX_POSITION_SIZE_PCT = 0.2 → 0.3 veya 0.4 yap (çok sinyal reddediliyor)")
        suggestions += 1
    if low_conf_rate > 50:
        report_lines.append("• Regime confidence çok düşük → hard_reject=0.30 → 0.20 düşür")
        suggestions += 1
    if profit_factor is not None and profit_factor < 1.3:
        report_lines.append(
            "• Strateji zarar ettiriyor → RSI threshold'ları gevşet veya regime ignore=True dene"
        )
        suggestions += 1
    if duration and total_trades == 0:
        report_lines.append("• Trade çok az → RSI_RANGE_OB ve RSI_RANGE_STR artır (10 → 15-20)")
        suggestions += 1
    if suggestions == 0:
        report_lines.append(
            "✓ Kayda değer bir sorun tespit edilmedi. Parametreler stabil görünüyor."
        )

    report_lines.append("\n📥 SIGNAL → TRADE FUNNEL")
    report_lines.append("-" * 50)
    signals_generated = len(re.findall(r"\[PROCESS\] Processing symbol:", content))
    report_lines.append(f"Üretilen sinyal adayları : {signals_generated}")
    report_lines.append(f"Gerçekleşen trade sayısı  : {total_trades}")

    if signals_generated > 0:
        conversion_rate = (total_trades / signals_generated) * 100
        report_lines.append(f"Sinyal → Trade dönüşümü : {conversion_rate:.1f}%")
    else:
        report_lines.append("Sinyal → Trade dönüşümü : N/A (sinyal adayı yok)")

    return "\n".join(report_lines)


# DISABLED: LogUploader moved to V1 model (LogUploader/__init__.py) using Azure SDK
# V2 model version below used subprocess with 'az' CLI which doesn't exist in function environment
# 
# @app.route(route="LogUploader", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
# def LogUploader(req: func.HttpRequest) -> func.HttpResponse:
#     ... (commented out - see LogUploader/__init__.py for working version)
