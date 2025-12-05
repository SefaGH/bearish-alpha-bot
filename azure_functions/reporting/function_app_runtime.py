import json
import os
import azure.functions as func
import logging
import re
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError, ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.kusto.data import KustoConnectionStringBuilder, KustoClient
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import RunCommandInput
from azure.storage.blob import (
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)
from jinja2 import Environment, FileSystemLoader, select_autoescape

app = func.FunctionApp()
LOGGER = logging.getLogger(__name__)

STORAGE_CONNECTION = os.environ.get("bearishstorage_STORAGE")
REPORTS_CONTAINER = os.environ.get("REPORTS_CONTAINER", "reports")

LOG_ANALYTICS_WORKSPACE_ID = os.environ.get("LOG_ANALYTICS_WORKSPACE_ID", "")
LOG_ANALYTICS_WORKSPACE_URL = os.environ.get(
    "LOG_ANALYTICS_WORKSPACE_URL", "https://api.loganalytics.io/v1"
)
ADX_CLUSTER_URI = os.environ.get("ADX_CLUSTER_URI", "")
ADX_DATABASE = os.environ.get("ADX_DATABASE", "")
REPORTS_STORAGE_ACCOUNT = os.environ.get("REPORTS_STORAGE_ACCOUNT", "bearishstorage")
RUN_REPORT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
REPORT_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(RUN_REPORT_TEMPLATE_DIR), autoescape=select_autoescape()
)

LOGUPLOADER_DEFAULT_VM = os.environ.get("LOGUPLOADER_VM_NAME", "BearishAlphaBot-VM-01")
LOGUPLOADER_DEFAULT_RG = os.environ.get("LOGUPLOADER_RESOURCE_GROUP", "TradeBot")
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "74ab10ba-c96d-449e-97cb-ee4f9c0de714")
VM_LOG_DIR = os.environ.get("LOGUPLOADER_VM_LOG_DIR", "/mnt/bearish/logs")
LOG_FILE_PATTERN = os.environ.get("LOGUPLOADER_FILE_GLOB", "live_trading_*.log")
STORAGE_ACCOUNT = os.environ.get("LOGUPLOADER_STORAGE_ACCOUNT", "bearishstorage")
RAW_LOG_CONTAINER = os.environ.get("RAW_LOGS_CONTAINER", "raw-logs")
STORAGE_API_VERSION = os.environ.get("LOGUPLOADER_STORAGE_API_VERSION", "2021-08-06")
RUN_COMMAND_TIMEOUT = int(os.environ.get("LOGUPLOADER_TIMEOUT_SECONDS", "180"))

BASH_TEMPLATE = f"""
set -e

LOG_DIR=\"{VM_LOG_DIR}\"
PATTERN=\"{LOG_FILE_PATTERN}\"
STORAGE_ACCOUNT=\"{STORAGE_ACCOUNT}\"
CONTAINER=\"{RAW_LOG_CONTAINER}\"

LATEST_LOG=$(ls -t \"$LOG_DIR\"/$PATTERN 2>/dev/null | head -1)

if [ -z \"$LATEST_LOG\" ]; then
    echo '{{"status":"error","message":"No log files found in {VM_LOG_DIR}"}}'
    exit 1
fi

FILENAME=$(basename \"$LATEST_LOG\")
CONTENT_LENGTH=$(stat -c%s \"$LATEST_LOG\")

TOKEN=$(curl -s -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://storage.azure.com/" | jq -r .access_token)

if [ -z \"$TOKEN\" ] || [ \"$TOKEN\" = \"null\" ]; then
    echo '{{"status":"error","message":"Failed to fetch managed identity token"}}'
    exit 1
fi

BLOB_URL=\"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{RAW_LOG_CONTAINER}/$FILENAME\"

HTTP_CODE=$(curl -X PUT \"$BLOB_URL\" \
    -H "Authorization: Bearer $TOKEN" \
    -H "x-ms-blob-type: BlockBlob" \
    -H "x-ms-version: {STORAGE_API_VERSION}" \
    -H "Content-Length: $CONTENT_LENGTH" \
    --data-binary \"@$LATEST_LOG\" \
    -s -o /dev/null -w \"%{{http_code}}\")

if [ \"$HTTP_CODE\" = \"201\" ]; then
    echo '{{"status":"success","file":"'$FILENAME'","size":'$CONTENT_LENGTH'}}'
else
    echo '{{"status":"error","message":"Upload failed","code":"'$HTTP_CODE'"}}'
    exit 1
fi
"""


@app.function_name(name="LogUploaderHttp")
@app.route(route="loguploader", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def log_uploader_http(req: func.HttpRequest) -> func.HttpResponse:
    body = _get_request_body(req)
    vm_name = body.get("vmName") or req.params.get("vmName") or LOGUPLOADER_DEFAULT_VM
    resource_group = body.get("resourceGroup") or req.params.get("resourceGroup") or LOGUPLOADER_DEFAULT_RG

    try:
        command_result = _invoke_vm_log_sync(vm_name, resource_group)
        vm_payload = _parse_vm_output(command_result) or {}
        status = vm_payload.get("status", "error")
        http_code = 200 if status == "success" else 500
        response_body = {
            "status": status,
            "vm": vm_name,
            "resourceGroup": resource_group,
            "vmPayload": vm_payload,
        }
        return func.HttpResponse(json.dumps(response_body), status_code=http_code, mimetype="application/json")
    except Exception as exc:  # noqa: BLE001
        logging.exception("LogUploader HTTP trigger failed")
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "vm": vm_name,
                    "resourceGroup": resource_group,
                    "message": str(exc),
                }
            ),
            status_code=500,
            mimetype="application/json",
        )

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


def _invoke_vm_log_sync(vm_name: str, resource_group: str):
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    compute_client = ComputeManagementClient(credential, SUBSCRIPTION_ID)
    command_input = RunCommandInput(command_id="RunShellScript", script=[BASH_TEMPLATE])
    poller = compute_client.virtual_machines.begin_run_command(
        resource_group_name=resource_group,
        vm_name=vm_name,
        parameters=command_input,
    )
    logging.info("Triggered RunCommand on %s/%s", resource_group, vm_name)
    return poller.result(timeout=RUN_COMMAND_TIMEOUT)


def _parse_vm_output(result) -> Dict[str, Any]:
    if not result.value:
        return {"status": "error", "message": "VM RunCommand returned no output"}

    message = result.value[0].message or ""
    logging.info("VM command raw output: %s", message[:2000])

    stdout_section = message
    if "[stdout]" in message and "[stderr]" in message:
        stdout_section = message.split("[stdout]", 1)[1].split("[stderr]", 1)[0].strip()
    stdout_section = stdout_section.replace("Enable succeeded:", "", 1).strip()

    try:
        return json.loads(stdout_section)
    except json.JSONDecodeError:
        logging.error("Failed to parse VM output JSON")
        return {"status": "error", "message": "Failed to parse VM output", "rawOutput": stdout_section[:500]}


def _get_request_body(req: func.HttpRequest) -> Dict[str, Any]:
    try:
        return req.get_json() or {}
    except ValueError:
        return {}


def _get_default_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential(exclude_shared_token_cache_credential=True)


def _query_events_with_fallback(run_id: str, credential: DefaultAzureCredential) -> list[dict[str, Any]]:
    if not LOG_ANALYTICS_WORKSPACE_ID:
        raise RuntimeError("LOG_ANALYTICS_WORKSPACE_ID is not configured")

    logs_client = LogsQueryClient(credential, endpoint=LOG_ANALYTICS_WORKSPACE_URL)
    query = f"BearishEvents_CL | where run_id_s == '{run_id}' | order by timestamp_utc_t asc"
    LOGGER.info("Executing Log Analytics query", extra={"run_id": run_id})
    try:
        result = logs_client.query_workspace(
            workspace_id=LOG_ANALYTICS_WORKSPACE_ID,
            query=query,
            timespan=timedelta(days=1),
        )
    except HttpResponseError as err:
        error_model = getattr(err, "error", None)
        error_code = getattr(error_model, "code", "") if error_model else ""
        inner_error = getattr(error_model, "innererror", None)
        inner_code = getattr(inner_error, "code", "") if inner_error else ""
        inner_message = getattr(inner_error, "message", "") if inner_error else ""
        error_text = " ".join(filter(None, [str(err), error_code, inner_code, inner_message]))

        if any(token in error_text for token in ("SemanticError", "SEM0100", "BearishEvents_CL")):
            LOGGER.warning(
                "BearishEvents_CL table missing; returning empty result",
                extra={"run_id": run_id},
            )
            return []
        raise

    if result.status == LogsQueryStatus.SUCCESS:
        return [row.to_dict() for row in result.tables[0].rows]

    LOGGER.warning(
        "Log Analytics query failed, falling back to ADX",
        extra={"status": result.status, "run_id": run_id},
    )

    if not ADX_CLUSTER_URI or not ADX_DATABASE:
        raise RuntimeError("ADX_CLUSTER_URI/ADX_DATABASE must be set for fallback")

    kcsb = KustoConnectionStringBuilder.with_aad_managed_service_identity(ADX_CLUSTER_URI)
    client = KustoClient(kcsb)
    response = client.execute(ADX_DATABASE, query)
    return [dict(row) for row in response.primary_results[0]]


def _render_report_html(events: list[dict[str, Any]]) -> str:
    template = REPORT_TEMPLATE_ENV.get_template("report.html.j2")
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }
    return template.render(context)


def _create_pdf_bytes(events: list[dict[str, Any]], run_id: str) -> bytes:
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Bearish Alpha Bot Report - Run {run_id}", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(f"Generated at: {datetime.now(timezone.utc).isoformat()}", styles["Normal"])
    )
    story.append(Spacer(1, 12))

    data = [["Timestamp", "Event Type", "Message", "Level"]]
    for event in events:
        ts = event.get("timestamp_utc_t") or event.get("TimeGenerated") or "N/A"
        etype = event.get("event_type_s") or "N/A"
        msg = event.get("message_s") or "N/A"
        level = event.get("level_s") or "INFO"
        msg = str(msg)
        if len(msg) > 50:
            msg = msg[:47] + "..."
        data.append([str(ts), str(etype), msg, str(level)])

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(table)
    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _upload_pdf(run_id: str, pdf_bytes: bytes, credential: DefaultAzureCredential) -> str:
    if not REPORTS_STORAGE_ACCOUNT or not REPORTS_CONTAINER:
        raise RuntimeError("REPORTS_STORAGE_ACCOUNT/REPORTS_CONTAINER must be configured")

    blob_client = BlobClient(
        account_url=f"https://{REPORTS_STORAGE_ACCOUNT}.blob.core.windows.net",
        container_name=REPORTS_CONTAINER,
        blob_name=f"{run_id}.pdf",
        credential=credential,
    )

    LOGGER.info("Uploading report to Blob Storage", extra={"blob": f"{run_id}.pdf"})
    blob_client.upload_blob(pdf_bytes, overwrite=True)

    service_client = blob_client.get_blob_service_client()
    ud_key = service_client.get_user_delegation_key(
        key_start_time=datetime.now(timezone.utc),
        key_expiry_time=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=blob_client.container_name,
        blob_name=blob_client.blob_name,
        user_delegation_key=ud_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    return f"{blob_client.url}?{sas_token}"


def _send_report_email(run_id: str, report_url: str) -> bool:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        LOGGER.warning("SENDGRID_API_KEY not found, skipping email dispatch")
        return False

    message = Mail(
        from_email="reports@bearish-bot.com",
        to_emails="sefaasar@hotmail.com",
        subject=f"Bearish Bot Report - Run {run_id}",
        html_content=(
            f"<h3>Trading Run Report</h3><p>Run ID: {run_id}</p>"
            f"<p><a href=\"{report_url}\">Download PDF Report</a></p>"
        ),
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        return 200 <= response.status_code < 300
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to send email", exc_info=exc)
        return False


def run_report_logic(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON", status_code=400)

    run_id = body.get("run_id")
    if not run_id:
        return func.HttpResponse("run_id is required", status_code=400)

    credential = _get_default_credential()

    try:
        events = _query_events_with_fallback(run_id, credential)
    except Exception as exc:  # noqa: BLE001
        error_details = traceback.format_exc()
        LOGGER.exception("Failed to retrieve events", exc_info=exc, extra={"run_id": run_id})
        return func.HttpResponse(
            json.dumps({"error": "Failed to retrieve events", "details": str(exc), "trace": error_details}),
            status_code=500,
            mimetype="application/json",
        )

    if not events:
        LOGGER.warning("No events found for run", extra={"run_id": run_id})
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "no_data",
                    "run_id": run_id,
                    "message": "No telemetry available yet; try again after a trading run.",
                }
            ),
            status_code=200,
            mimetype="application/json",
        )

    try:
        pdf_bytes = _create_pdf_bytes(events, run_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to generate PDF", exc_info=exc, extra={"run_id": run_id})
        return func.HttpResponse("Failed to generate PDF", status_code=500)

    try:
        sas_url = _upload_pdf(run_id, pdf_bytes, credential)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to upload report", exc_info=exc, extra={"run_id": run_id})
        return func.HttpResponse("Failed to upload report", status_code=500)

    email_sent = _send_report_email(run_id, sas_url)

    response_body = {
        "run_id": run_id,
        "report_url": sas_url,
        "email_sent": email_sent,
    }

    return func.HttpResponse(json.dumps(response_body), status_code=200, mimetype="application/json")


@app.function_name(name="run_report")
@app.route(route="run_report", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_report(req: func.HttpRequest) -> func.HttpResponse:
    return run_report_logic(req)
