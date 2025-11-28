# Reporting Function Deployment Guide

1. **Prerequisites**
   - Run `infra/step05-function-logic.ps1` after core resources (steps 01–04).
   - Ensure Key Vault contains secrets:
     - `log-analytics-workspace-id`
     - `log-analytics-shared-key`
     - (Optional) `sendgrid-api-key` or equivalent email credential.
   - Confirm Function MSI has `Log Analytics Reader` and `Storage Blob Data Contributor` roles (script assigns them).

2. **Local Validation**
   - Install requirements in a virtual environment:
     ```bash
     cd azure_functions/reporting
     python -m venv .venv
     source .venv/bin/activate  # Windows: .venv\Scripts\activate
     pip install -r requirements.txt
     func start
     ```
   - Send test request:
     ```bash
     curl -X POST http://localhost:7071/api/run-report -H "Content-Type: application/json" -d '{"run_id":"demo"}'
     ```

3. **Publishing**
   - Use Azure Functions Core Tools:
     ```bash
     func azure functionapp publish bearish-reporting-func --python
     ```
   - Alternatively configure CI/CD (GitHub Actions) targeting Python 3.11 with `func publish`.

4. **Email / PDF Integration**
   - Replace placeholder upload logic in `run_report/__init__.py` with Playwright/ReportLab rendering.
   - Use `generate_blob_sas` to produce time-limited SAS links for emailed reports.
   - Integrate SendGrid or Logic App connector for notifications.

5. **Logic App Trigger**
   - Logic App (`bearish-report-orchestrator`) is created with manual HTTP trigger; connect Container Insights alert or schedule after validation.
   - Update template or workflow to pass additional metadata (duration, severity).

6. **Monitoring**
   - Enable Application Insights for the Function App (consumption plan includes it by default).
   - Use `az monitor app-insights component create` if dedicated workspace required.
