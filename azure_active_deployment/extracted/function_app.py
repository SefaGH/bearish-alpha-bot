import azure.functions as func
import json
import logging
import traceback

# Only define the app globally; runtime logic stays lazily imported in the handler.
app = func.FunctionApp()


@app.route(route="run_report", auth_level=func.AuthLevel.FUNCTION)
def run_report(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Wrapper: run_report triggered.")

    try:
        # Lazy import ensures heavy dependencies load only when needed.
        import function_app_runtime as runtime

        return runtime.run_report_logic(req)
    except Exception as exc:  # noqa: BLE001 broad to surface import/runtime faults
        error_trace = traceback.format_exc()
        logging.error("CRITICAL WRAPPER ERROR: %s", error_trace)
        return func.HttpResponse(
            body=json.dumps(
                {
                    "status": "fatal_error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": error_trace,
                }
            ),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="loguploader", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def log_uploader(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Wrapper: log_uploader triggered.")

    try:
        # Lazy import ensures heavy dependencies load only when needed.
        import function_app_runtime as runtime

        return runtime.log_uploader_http(req)
    except Exception as exc:  # noqa: BLE001 broad to surface import/runtime faults
        error_trace = traceback.format_exc()
        logging.error("CRITICAL WRAPPER ERROR: %s", error_trace)
        return func.HttpResponse(
            body=json.dumps(
                {
                    "status": "fatal_error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": error_trace,
                }
            ),
            status_code=500,
            mimetype="application/json",
        )


__all__ = ["app"]
