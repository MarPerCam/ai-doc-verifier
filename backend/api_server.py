# backend/api_server.py
"""
Flask API Server for AI-Powered Document Verification
- Upload documents
- Extract with Gemini (via ai_backend_gemini.py)
- Cache per document + cache per workflow (SQLite)
- Reverify: clears ONLY caches for current uploaded docs + current workflow, then reprocesses
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ai_backend_gemini import (
    AIDocumentExtractor,
    CNPJValidator,
    DocumentComparator,
    DocumentData,
)
from database import (
    delete_document_cache_by_hash,
    delete_workflow_cache,
    get_document_cache,
    get_workflow_cache,
    init_db,
    save_document_cache,
    save_workflow_cache,
)
from utils import sha256_file, workflow_hash

load_dotenv()

# -----------------------------------------------------------------------------
# Paths / Config
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
LOGS_FOLDER = BASE_DIR / "logs"

ALLOWED_EXTENSIONS = {"pdf", "xlsx", "xls", "jpg", "jpeg", "png"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
LOGS_FOLDER.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_FOLDER / "api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("api")

# -----------------------------------------------------------------------------
# Init DB + AI
# -----------------------------------------------------------------------------

init_db()

try:
    ai_extractor = AIDocumentExtractor()
    logger.info("✅ Gemini READY")
except Exception as e:
    ai_extractor = None
    logger.exception("❌ Failed to initialize AI extractor: %s", e)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file_storage, doc_type: str) -> Path:
    filename = secure_filename(file_storage.filename or "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{ts}_{doc_type}_{filename}"
    path = UPLOAD_FOLDER / unique_name
    file_storage.save(path)
    return path


def _as_dict(doc: DocumentData) -> dict:
    return {
        "shipper_name": doc.shipper_name,
        "consignee": doc.consignee,
        "cnpj": doc.cnpj,
        "localization": doc.localization,
        "ncm_4d": getattr(doc, "ncm_4d", None),
        "ncm_8d": getattr(doc, "ncm_8d", None),
        "packages": doc.packages,
        "gross_weight": doc.gross_weight,
        "cbm": doc.cbm,
        "extraction_method": doc.extraction_method,
        "confidence": doc.confidence,
    }


def _extract_with_cache(filepath: str, doc_type: str, force: bool) -> dict:
    file_hash = sha256_file(filepath)

    if not force:
        cached = get_document_cache(file_hash, doc_type)
        if cached:
            logger.info("📦 Document cache HIT (%s)", doc_type)
            return cached

    logger.info("🤖 Extracting (%s) force=%s", doc_type, force)
    data = ai_extractor.extract_from_file(filepath, doc_type)
    data_dict = _as_dict(data)

    save_document_cache(file_hash, doc_type, Path(filepath).name, data_dict)
    return data_dict


def _build_report(files: Dict[str, str], extracted_data: Dict[str, dict]) -> Dict[str, Any]:
    cnpj_validation = None
    cnpj = extracted_data.get("bl", {}).get("cnpj") or extracted_data.get("invoice", {}).get("cnpj")
    if cnpj:
        cnpj_validation = CNPJValidator().validate_online(cnpj)

    bl_obj = DocumentData(**extracted_data["bl"])
    inv_obj = DocumentData(**extracted_data["invoice"])
    pk_obj = DocumentData(**extracted_data["packing"]) if "packing" in extracted_data else None

    comparison = DocumentComparator().compare(bl_obj, inv_obj, pk_obj)

    total = comparison.get("total_checks", 0)
    passed = comparison.get("passed", 0)

    return {
        "timestamp": now_iso(),
        "documents_processed": list(files.keys()),
        "extracted_data": extracted_data,
        "cnpj_validation": cnpj_validation,
        "comparison": comparison,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": comparison.get("failed", 0),
            "success_rate": f"{(passed / total * 100):.1f}%" if total else "N/A",
        },
    }


def _save_report_file(report: Dict[str, Any]) -> str:
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = OUTPUT_FOLDER / report_filename
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_filename


def _collect_files_from_request() -> Dict[str, str]:
    files: Dict[str, str] = {}
    for key in ["bl", "invoice", "packing"]:
        if key in request.files:
            f = request.files[key]
            if f and f.filename:
                path = save_uploaded_file(f, key)
                files[key] = str(path)
                logger.info("Saved %s: %s", key, path.name)
    return files


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def home():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>ai-doc-verifier</title>
  </head>
  <body style="font-family: Arial; padding: 16px;">
    <h2>ai-doc-verifier API online ✅</h2>
    <p>Este servidor é a API. O frontend roda separado (porta 5500).</p>
    <ul>
      <li><a href="/api/health">/api/health</a></li>
      <li>POST /api/process-complete</li>
      <li>POST /api/reverify</li>
    </ul>
  </body>
</html>
""", 200


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return "", 204

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "timestamp": now_iso(),
            "ai_enabled": ai_extractor is not None,
            "version": "3.0.0",
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        doc_type = request.form.get("doc_type", "unknown")

        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": f"File type not allowed. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}
            ), 400

        filepath = save_uploaded_file(file, doc_type)

        return jsonify(
            {
                "success": True,
                "filename": filepath.name,
                "doc_type": doc_type,
                "size": filepath.stat().st_size,
                "path": str(filepath),
            }
        ), 200

    except Exception as e:
        logger.exception("Upload error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/extract", methods=["POST"])
def extract():
    try:
        if not ai_extractor:
            return jsonify({"error": "AI extractor not initialized. Check GEMINI_API_KEY"}), 500

        payload = request.get_json(silent=True) or {}
        filepath = payload.get("filepath")
        doc_type = payload.get("doc_type", "unknown")
        force = bool(payload.get("force", False))

        if not filepath:
            return jsonify({"error": "No filepath provided"}), 400

        path = Path(filepath)
        if not path.exists():
            return jsonify({"error": f"File not found: {filepath}"}), 404

        data_dict = _extract_with_cache(str(path), doc_type, force=force)
        return jsonify({"success": True, "doc_type": doc_type, "data": data_dict}), 200

    except Exception as e:
        logger.exception("Extraction error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/process-complete", methods=["POST"])
def process_complete():
    try:
        if not ai_extractor:
            return jsonify({"error": "AI extractor not initialized. Check GEMINI_API_KEY"}), 500

        force = request.args.get("force", "0") == "1"

        files = _collect_files_from_request()
        if "bl" not in files or "invoice" not in files:
            return jsonify({"error": "At least BL and Invoice are required"}), 400

        bl_h = sha256_file(files["bl"])
        inv_h = sha256_file(files["invoice"])
        pk_h = sha256_file(files["packing"]) if "packing" in files else None
        wf_h = workflow_hash(bl_h, inv_h, pk_h)

        if not force:
            cached = get_workflow_cache(wf_h)
            if cached:
                logger.info("Workflow cache HIT — returning stored report")
                return jsonify(
                    {
                        "success": True,
                        "report": cached,
                        "cached": True,
                        "workflow_hash": wf_h,
                        "files": files,
                    }
                ), 200

        if force:
            delete_workflow_cache(wf_h)

        extracted_data: Dict[str, dict] = {
            "bl": _extract_with_cache(files["bl"], "bl", force=force),
            "invoice": _extract_with_cache(files["invoice"], "invoice", force=force),
        }
        if "packing" in files:
            extracted_data["packing"] = _extract_with_cache(files["packing"], "packing", force=force)

        report = _build_report(files, extracted_data)

        save_workflow_cache(wf_h, bl_h, inv_h, pk_h, report)
        report_filename = _save_report_file(report)

        return jsonify(
            {
                "success": True,
                "report": report,
                "report_file": report_filename,
                "cached": False,
                "workflow_hash": wf_h,
                "files": files,
                "forced": force,
            }
        ), 200

    except Exception as e:
        logger.exception("Workflow failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reverify", methods=["POST"])
def reverify():
    """
    Recebe novamente os arquivos, apaga SOMENTE caches dos hashes desses arquivos
    e do workflow atual, e reprocessa com Gemini (force=True).
    """
    try:
        if not ai_extractor:
            return jsonify({"error": "AI extractor not initialized. Check GEMINI_API_KEY"}), 500

        files = _collect_files_from_request()
        if "bl" not in files or "invoice" not in files:
            return jsonify({"error": "At least BL and Invoice are required"}), 400

        bl_h = sha256_file(files["bl"])
        inv_h = sha256_file(files["invoice"])
        pk_h = sha256_file(files["packing"]) if "packing" in files else None
        wf_h = workflow_hash(bl_h, inv_h, pk_h)

        delete_document_cache_by_hash(bl_h)
        delete_document_cache_by_hash(inv_h)
        if pk_h:
            delete_document_cache_by_hash(pk_h)

        delete_workflow_cache(wf_h)

        extracted_data: Dict[str, dict] = {
            "bl": _extract_with_cache(files["bl"], "bl", force=True),
            "invoice": _extract_with_cache(files["invoice"], "invoice", force=True),
        }
        if "packing" in files:
            extracted_data["packing"] = _extract_with_cache(files["packing"], "packing", force=True)

        report = _build_report(files, extracted_data)

        save_workflow_cache(wf_h, bl_h, inv_h, pk_h, report)
        report_filename = _save_report_file(report)

        return jsonify(
            {
                "success": True,
                "reverified": True,
                "report": report,
                "report_file": report_filename,
                "cached": False,
                "workflow_hash": wf_h,
                "files": files,
            }
        ), 200

    except Exception as e:
        logger.exception("Reverify failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reports", methods=["GET"])
def list_reports():
    try:
        reports = []
        for file in OUTPUT_FOLDER.glob("*.json"):
            reports.append(
                {
                    "filename": file.name,
                    "created": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "size": file.stat().st_size,
                }
            )
        reports.sort(key=lambda x: x["created"], reverse=True)
        return jsonify({"success": True, "reports": reports, "count": len(reports)}), 200
    except Exception as e:
        logger.exception("List reports error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reports/<filename>", methods=["GET"])
def download_report(filename: str):
    try:
        return send_from_directory(str(OUTPUT_FOLDER), filename, as_attachment=True)
    except Exception:
        return jsonify({"error": "Report not found"}), 404


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"}), 413


if __name__ == "__main__":
    print("\nAI DOCUMENT VERIFIER READY")
    print("http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
