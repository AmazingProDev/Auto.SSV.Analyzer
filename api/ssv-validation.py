import sys
import os
import json
from http import HTTPStatus

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import parse_multipart_form_data
from ssv_validation.service import validate_ssv_workbook, SsvValidationError

def app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "")
    headers = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ]
    
    if method == "OPTIONS":
        start_response("204 No Content", headers)
        return [b""]
        
    if method == "POST":
        try:
            content_length = int(environ.get("CONTENT_LENGTH", "0"))
        except ValueError:
            content_length = 0
            
        if content_length <= 0:
            start_response("400 Bad Request", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": "No upload payload was received."}).encode("utf-8")]
            
        content_type = environ.get("CONTENT_TYPE", "")
        if "multipart/form-data" not in content_type:
            start_response("400 Bad Request", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": "The upload must use multipart/form-data."}).encode("utf-8")]
            
        body = environ["wsgi.input"].read(content_length)
        
        try:
            fields = parse_multipart_form_data(content_type, body)
        except ValueError as exc:
            start_response("400 Bad Request", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": str(exc)}).encode("utf-8")]
            
        uploaded_file = None
        for field_parts in fields.values():
            for part in field_parts:
                if part.get("filename"):
                    uploaded_file = part
                    break
            if uploaded_file:
                break
                
        if not uploaded_file:
            start_response("400 Bad Request", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": "No Excel file was uploaded."}).encode("utf-8")]
            
        filename = uploaded_file.get("filename") or "upload.xlsx"
        if not filename.lower().endswith(".xlsx"):
            start_response("400 Bad Request", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": "Invalid file format. Please upload an .xlsx workbook."}).encode("utf-8")]
            
        include_all_previews = False
        include_preview_parts = fields.get("includeAllPreviews", [])
        if include_preview_parts:
            val = include_preview_parts[0].get("data", b"").decode("utf-8", "ignore").strip().lower()
            if val in {"1", "true", "yes", "on"}:
                include_all_previews = True
            
        try:
            response = validate_ssv_workbook(uploaded_file["data"], filename, include_all_previews=include_all_previews)
        except SsvValidationError as exc:
            start_response("400 Bad Request", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": str(exc)}).encode("utf-8")]
        except Exception as exc:
            start_response("500 Internal Server Error", headers + [("Content-Type", "application/json")])
            return [json.dumps({"success": False, "error": f"Unexpected server error: {exc}"}).encode("utf-8")]
            
        start_response("200 OK", headers + [("Content-Type", "application/json")])
        return [json.dumps(response).encode("utf-8")]

    start_response("405 Method Not Allowed", headers + [("Content-Type", "application/json")])
    return [json.dumps({"success": False, "error": f"Method {method} not allowed on endpoint /api/ssv-validation. Please use POST."}).encode("utf-8")]
