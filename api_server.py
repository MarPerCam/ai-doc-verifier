"""
Flask API Server for AI-Powered Document Verification
- Upload documents
- Extract with Gemini (via ai_backend_gemini.py)
- Cache per document + cache per workflow
- Force re-verify: bypass caches and overwrite DB
"""
def save_uploaded_file(file, doc_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{doc_type}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath

from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path


import database
from database import (
    init_db,
    get_document_cache,
    save_document_cache,
    get_workflow_cache,
    save_workflow_cache,
    delete_workflow_cache,
    delete_document_cache_by_hash 
)

from utils import sha256_file, workflow_hash

from ai_backend_gemini import (
    AIDocumentExtractor,
    CNPJValidator,
    DocumentComparator,
    DocumentData,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)



if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# -----------------------------------------------------------------------------
# App + folders
# -----------------------------------------------------------------------------


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def save_uploaded_file(file, doc_type):
    filename = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(UPLOAD_FOLDER, f"{ts}_{doc_type}_{filename}")
    file.save(path)
    return path


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = Path(__file__).parent.parent / "outputs"
LOGS_FOLDER = Path(__file__).parent.parent / "logs"

ALLOWED_EXTENSIONS = {"pdf", "xlsx", "xls", "jpg", "jpeg", "png"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
LOGS_FOLDER.mkdir(exist_ok=True, parents=True)

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


def _as_dict(doc: DocumentData) -> dict:
    return {
        "shipper_name": doc.shipper_name,
        "consignee": doc.consignee,
        "cnpj": doc.cnpj,
        "localization": doc.localization,
        "ncm": doc.ncm,
        "packages": doc.packages,
        "gross_weight": doc.gross_weight,
        "cbm": doc.cbm,
        "extraction_method": doc.extraction_method,
        "confidence": doc.confidence,
    }


def _extract_with_cache(filepath: str, doc_type: str, force: bool) -> dict:
    """
    Retorna dict com os campos extraídos.
    - Se force=False: usa cache se existir
    - Se force=True: ignora cache e sobrescreve cache
    """
    fh = sha256_file(filepath)

    if not force:
        cached = get_document_cache(fh, doc_type)
        if cached:
            logger.info(f"📦 Document cache HIT ({doc_type})")
            return cached

    logger.info(f"🤖 Extracting ({doc_type}) force={force}")
    data = ai_extractor.extract_from_file(filepath, doc_type)
    data_dict = _as_dict(data)

    # sobrescreve cache
    save_document_cache(fh, doc_type, Path(filepath).name, data_dict)
    return data_dict


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "timestamp": now_iso(),
        "ai_enabled": ai_extractor is not None,
        "version": "2.0.0"
    })


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
            return jsonify({"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{ts}_{doc_type}_{filename}"
        filepath = UPLOAD_FOLDER / unique_name

        file.save(filepath)

        logger.info(f"Saved upload: {unique_name} ({doc_type})")

        return jsonify({
            "success": True,
            "filename": unique_name,
            "original_name": filename,
            "doc_type": doc_type,
            "size": filepath.stat().st_size,
            "path": str(filepath),
        }), 200

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
        if not os.path.exists(filepath):
            return jsonify({"error": f"File not found: {filepath}"}), 404

        data_dict = _extract_with_cache(filepath, doc_type, force=force)

        return jsonify({
            "success": True,
            "doc_type": doc_type,
            "data": data_dict,
        }), 200

    except Exception as e:
        logger.exception("Extraction error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/process-complete", methods=["POST"])
def process_complete():
    """
    Multipart: bl, invoice, packing(optional)
    Query param:
      - force=1 -> ignora caches, recalcula e SOBRESCREVE caches
    """
    try:
        if not ai_extractor:
            return jsonify({"error": "AI extractor not initialized. Check GEMINI_API_KEY"}), 500

        force = request.args.get("force", "0") == "1"

        # salvar arquivos
        files = {}
        for key in ["bl", "invoice", "packing"]:
            if key in request.files:
                f = request.files[key]
                if f and f.filename:
                    filename = secure_filename(f.filename)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_name = f"{ts}_{key}_{filename}"
                    path = UPLOAD_FOLDER / unique_name
                    f.save(path)
                    files[key] = str(path)
                    logger.info(f"Saved {key}: {unique_name}")

        if "bl" not in files or "invoice" not in files:
            return jsonify({"error": "At least BL and Invoice are required"}), 400

        # hashes por conteúdo
        bl_h = sha256_file(files["bl"])
        inv_h = sha256_file(files["invoice"])
        pk_h = sha256_file(files["packing"]) if "packing" in files else None
        wf_h = workflow_hash(bl_h, inv_h, pk_h)

        # workflow cache
        if not force:
            cached = get_workflow_cache(wf_h)
            if cached:
                logger.info("Workflow cache HIT — returning stored report")
                return jsonify({
                    "success": True,
                    "report": cached,
                    "cached": True,
                    "workflow_hash": wf_h,
                    "files": files,   # ajuda o frontend a reverificar sem reupload
                }), 200

        # force -> apaga workflow cache anterior (para não “voltar”)
        if force:
            delete_workflow_cache(wf_h)

        # extrair documentos (com/sem cache conforme force)
        extracted_data = {}
        extracted_data["bl"] = _extract_with_cache(files["bl"], "bl", force=force)
        extracted_data["invoice"] = _extract_with_cache(files["invoice"], "invoice", force=force)
        if "packing" in files:
            extracted_data["packing"] = _extract_with_cache(files["packing"], "packing", force=force)

        # validar cnpj
        cnpj_validation = None
        cnpj = extracted_data.get("bl", {}).get("cnpj") or extracted_data.get("invoice", {}).get("cnpj")
        if cnpj:
            cnpj_validation = CNPJValidator().validate_online(cnpj)

        # comparar
        bl_obj = DocumentData(**extracted_data["bl"])
        inv_obj = DocumentData(**extracted_data["invoice"])
        pk_obj = DocumentData(**extracted_data["packing"]) if "packing" in extracted_data else None

        comparison = DocumentComparator().compare(bl_obj, inv_obj, pk_obj)

        report = {
            "timestamp": now_iso(),
            "documents_processed": list(files.keys()),
            "extracted_data": extracted_data,
            "cnpj_validation": cnpj_validation,
            "comparison": comparison,
            "summary": {
                "total_checks": comparison.get("total_checks", 0),
                "passed": comparison.get("passed", 0),
                "failed": comparison.get("failed", 0),
                "success_rate": (
                    f"{(comparison['passed'] / comparison['total_checks'] * 100):.1f}%"
                    if comparison.get("total_checks")
                    else "N/A"
                )
            }
        }

        # salva workflow cache
        save_workflow_cache(wf_h, bl_h, inv_h, pk_h, report)

        # salva report file
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = OUTPUT_FOLDER / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return jsonify({
            "success": True,
            "report": report,
            "report_file": report_filename,
            "cached": False,
            "workflow_hash": wf_h,
            "files": files,
            "forced": force,
        }), 200

    except Exception as e:
        logger.exception("Workflow failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reverify", methods=["POST"])
def reverify_documents():

    logging.info("🔄 Starting reverification")

    files = request.files

    extracted = {}

    for doc_type in ["bl", "invoice", "packing"]:
        if doc_type not in files:
            continue

        file = files[doc_type]

        filepath = save_uploaded_file(file, doc_type)

        file_hash = sha256_file(filepath)

        # limpa SOMENTE cache desse arquivo
        database.delete_document_cache_by_hash(file_hash)

        logging.info(f"♻ Cache cleared for {doc_type}")

        extracted[doc_type] = extractor.extract_from_file(filepath, doc_type)

        save_document_cache(file_hash, doc_type, extracted[doc_type])

    report = build_report(extracted)

    return jsonify({
        "success": True,
        "report": report
    })



@app.route("/api/reports", methods=["GET"])
def list_reports():
    try:
        reports = []
        for file in OUTPUT_FOLDER.glob("*.json"):
            reports.append({
                "filename": file.name,
                "created": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                "size": file.stat().st_size,
            })
        reports.sort(key=lambda x: x["created"], reverse=True)
        return jsonify({"success": True, "reports": reports, "count": len(reports)}), 200
    except Exception as e:
        logger.exception("List reports error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reports/<filename>", methods=["GET"])
def download_report(filename):
    try:
        return send_from_directory(str(OUTPUT_FOLDER), filename, as_attachment=True)
    except Exception:
        return jsonify({"error": "Report not found"}), 404


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"}), 413


if __name__ == "__main__":
    print()
    print("AI DOCUMENT VERIFIER READY")
    print("http://localhost:5000")
    print()

    app.run(host="0.0.0.0", port=5000, debug=True)
