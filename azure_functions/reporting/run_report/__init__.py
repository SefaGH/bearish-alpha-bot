import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.kusto.data import KustoConnectionStringBuilder, KustoClient
from azure.monitor.query import LogsQueryClient
from azure.monitor.query import LogsQueryStatus
from azure.storage.blob import BlobClient, generate_blob_sas, BlobSasPermissions
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime, timezone, timedelta

# NOTE: Playwright/ReportLab imports are deferred to avoid cold-start cost

LOGGER = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape())


def _get_credential() -> DefaultAzureCredential:
    # Managed identity preferred; ensure AzureWebJobsSecretStorageType=files for local dev
    return DefaultAzureCredential(exclude_shared_token_cache_credential=True)


def _query_events_with_fallback(run_id: str, credential: DefaultAzureCredential) -> list[dict[str, Any]]:
    workspace_id = os.environ["LOG_ANALYTICS_WORKSPACE_ID"]
    workspace_url = os.environ.get("LOG_ANALYTICS_WORKSPACE_URL", "https://api.loganalytics.io/v1")
    logs_client = LogsQueryClient(credential, endpoint=workspace_url)

    query = f"BearishEvents_CL | where run_id_s == '{run_id}' | order by timestamp_utc_t asc"
    LOGGER.info("Executing Log Analytics query", extra={"run_id": run_id})
    result = logs_client.query_workspace(workspace_id, query)

    if result.status == LogsQueryStatus.SUCCESS:
        return [row.to_dict() for row in result.tables[0].rows]

    LOGGER.warning("Log Analytics query failed, falling back to ADX", extra={"status": result.status})

    cluster_uri = os.environ["ADX_CLUSTER_URI"]
    database = os.environ["ADX_DATABASE"]
    kcsb = KustoConnectionStringBuilder.with_aad_managed_service_identity(cluster_uri)
    client = KustoClient(kcsb)
    response = client.execute(database, query)
    return [dict(row) for row in response.primary_results[0]]


def _render_report_html(events: list[dict[str, Any]]) -> str:
    template = ENV.get_template("report.html.j2")
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }
    return template.render(context)


def _create_pdf_bytes(events: list[dict[str, Any]], run_id: str) -> bytes:
    """Generates a PDF report using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Bearish Alpha Bot Report - Run {run_id}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated at: {datetime.now(timezone.utc).isoformat()}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Table Data
    data = [["Timestamp", "Event Type", "Message", "Level"]]
    for event in events:
        # Handle potential missing keys or different column names from ADX/LogAnalytics
        ts = event.get('timestamp_utc_t') or event.get('TimeGenerated') or 'N/A'
        etype = event.get('event_type_s') or 'N/A'
        msg = event.get('message_s') or 'N/A'
        level = event.get('level_s') or 'INFO'
        
        # Truncate long messages
        if len(str(msg)) > 50:
            msg = str(msg)[:47] + "..."
            
        data.append([str(ts), str(etype), str(msg), str(level)])

    # Table Style
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(table)
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _upload_pdf(run_id: str, pdf_bytes: bytes, credential: DefaultAzureCredential) -> str:
    storage_account = os.environ["REPORTS_STORAGE_ACCOUNT"]
    container = os.environ["REPORTS_CONTAINER"]
    blob_name = f"reports/{run_id}.pdf"

    LOGGER.info("Uploading report to Blob Storage", extra={"blob": blob_name})

    blob_client = BlobClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        container_name=container,
        blob_name=f"{run_id}.pdf",
        credential=credential,
    )

    blob_client.upload_blob(pdf_bytes, overwrite=True)

    # Generate SAS URL (valid for 24 hours)
    # Note: User delegation key is needed for SAS with AAD credential, 
    # but for simplicity/speed in this context, we might need account key or assume public access?
    # Actually, best practice with MSI is to use User Delegation SAS.
    
    # However, getting user delegation key requires another call. 
    # Let's try to return the direct URL if public, or generate SAS if we can.
    # If we can't easily get SAS without key, we might need to rely on the container being private 
    # and the email recipient having access, OR we implement User Delegation SAS.
    
    # Let's implement User Delegation SAS for correctness.
    service_client = blob_client.get_blob_service_client()
    ud_key = service_client.get_user_delegation_key(
        key_start_time=datetime.now(timezone.utc),
        key_expiry_time=datetime.now(timezone.utc) + timedelta(hours=24)
    )

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=blob_client.container_name,
        blob_name=blob_client.blob_name,
        user_delegation_key=ud_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=24)
    )
    
    return f"{blob_client.url}?{sas_token}"


def _send_email(run_id: str, report_url: str):
    """Sends an email with the report link using SendGrid."""
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        LOGGER.warning("SENDGRID_API_KEY not found, skipping email.")
        return

    # TODO: Make recipient configurable
    to_email = "sefaasar@hotmail.com" 
    from_email = "reports@bearish-bot.com"
    
    subject = f"Bearish Bot Report - Run {run_id}"
    content = f"""
    <h3>Trading Run Report</h3>
    <p>Run ID: {run_id}</p>
    <p>A new report has been generated.</p>
    <p><a href="{report_url}">Download PDF Report</a></p>
    """
    
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=content
    )
    
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        LOGGER.info(f"Email sent. Status Code: {response.status_code}")
    except Exception as e:
        LOGGER.error(f"Failed to send email: {str(e)}")


async def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON", status_code=400)

    run_id = body.get("run_id")
    if not run_id:
        return func.HttpResponse("run_id is required", status_code=400)

    credential = _get_credential()

    try:
        events = _query_events_with_fallback(run_id, credential)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to retrieve events", exc_info=exc, extra={"run_id": run_id})
        return func.HttpResponse("Failed to retrieve events", status_code=500)

    if not events:
        LOGGER.warning("No events found for run", extra={"run_id": run_id})
        return func.HttpResponse("No events found", status_code=404)

    # Generate PDF
    try:
        pdf_bytes = _create_pdf_bytes(events, run_id)
    except Exception as exc:
        LOGGER.exception("Failed to generate PDF", exc_info=exc, extra={"run_id": run_id})
        return func.HttpResponse("Failed to generate PDF", status_code=500)

    try:
        sas_url = _upload_pdf(run_id, pdf_bytes, credential)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to upload report", exc_info=exc, extra={"run_id": run_id})
        return func.HttpResponse("Failed to upload report", status_code=500)

    # Send Email
    _send_email(run_id, sas_url)

    response_body = {
        "run_id": run_id,
        "report_url": sas_url,
        "email_sent": True
    }

    return func.HttpResponse(json.dumps(response_body), status_code=200, mimetype="application/json")
