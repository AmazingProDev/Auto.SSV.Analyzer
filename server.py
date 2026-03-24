from __future__ import annotations

import json
import logging
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ssv_validation.service import SsvValidationError, validate_ssv_workbook

LOGGER = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent


def parse_multipart_form_data(content_type: str, body: bytes) -> dict[str, list[dict[str, Any]]]:
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        raise ValueError("Missing multipart boundary.")

    boundary = boundary_match.group(1).encode("utf-8")
    delimiter = b"--" + boundary
    fields: dict[str, list[dict[str, Any]]] = {}

    for chunk in body.split(delimiter):
        part = chunk.strip()
        if not part or part == b"--":
            continue

        if part.startswith(b"--"):
            part = part[2:]

        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue

        headers = {}
        for line in header_blob.decode("utf-8", "ignore").split("\r\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        disposition = headers.get("content-disposition", "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        if not name_match:
            continue

        name = name_match.group(1)
        payload = content[:-2] if content.endswith(b"\r\n") else content
        fields.setdefault(name, []).append(
            {
                "filename": filename_match.group(1) if filename_match else None,
                "content_type": headers.get("content-type", "application/octet-stream"),
                "data": payload,
            }
        )

    return fields


class AppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/ssv-validation":
            self.send_json(HTTPStatus.NOT_FOUND, {"success": False, "error": "Unknown API route."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid request length."})
            return

        if content_length <= 0:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "No upload payload was received."})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "The upload must use multipart/form-data."})
            return

        body = self.rfile.read(content_length)

        try:
            fields = parse_multipart_form_data(content_type, body)
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
            return

        uploaded_file = None
        for field_parts in fields.values():
            for part in field_parts:
                if part.get("filename"):
                    uploaded_file = part
                    break
            if uploaded_file:
                break

        if not uploaded_file:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "No Excel file was uploaded."})
            return

        filename = uploaded_file.get("filename") or "upload.xlsx"
        if not filename.lower().endswith(".xlsx"):
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid file format. Please upload an .xlsx workbook."})
            return

        include_all_previews = False
        include_preview_parts = fields.get("includeAllPreviews", [])
        if include_preview_parts:
            include_all_previews = (
                include_preview_parts[0].get("data", b"").decode("utf-8", "ignore").strip().lower() in {"1", "true", "yes", "on"}
            )

        try:
            response = validate_ssv_workbook(uploaded_file["data"], filename, include_all_previews=include_all_previews)
        except SsvValidationError as exc:
            LOGGER.warning("SSV validation failed: %s", exc)
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - API safeguard
            LOGGER.exception("Unexpected SSV validation error")
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": f"Unexpected server error: {exc}"})
            return

        self.send_json(HTTPStatus.OK, response)

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppRequestHandler)
    LOGGER.info("Serving 360 view app on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
