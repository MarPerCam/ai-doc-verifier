import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


DB_PATH = "cache.db"
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "90"))


def _db_file() -> Path:
    base = Path(__file__).parent
    if DB_PATH.startswith("./"):
        return base / DB_PATH
    return base / DB_PATH


def get_conn():
    db = _db_file()
    db.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db))


def _now_iso() -> str:
    return datetime.now().isoformat()


def _is_expired(created_at_str: str) -> bool:
    if not TTL_DAYS:
        return False
    try:
        created = datetime.fromisoformat(created_at_str)
        return datetime.now() > created + timedelta(days=TTL_DAYS)
    except Exception:
        return False


def _ensure_schema(conn: sqlite3.Connection):
    """
    Cria tabelas se não existirem e faz migração leve se DB antigo existir.
    """
    cur = conn.cursor()

    # document_cache
    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        filename TEXT,
        extracted_json TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # índice único
    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_unique
    ON document_cache(file_hash, doc_type)
    """)

    # workflow_cache
    cur.execute("""
    CREATE TABLE IF NOT EXISTS workflow_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_hash TEXT NOT NULL UNIQUE,
        bl_hash TEXT NOT NULL,
        invoice_hash TEXT NOT NULL,
        packing_hash TEXT,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_access TEXT NOT NULL
    )
    """)

    # MIGRAÇÃO: se alguém tinha um schema antigo sem extracted_json,
    # garantimos que a coluna exista.
    cur.execute("PRAGMA table_info(document_cache)")
    cols = {row[1] for row in cur.fetchall()}  # row[1] = name
    if "extracted_json" not in cols:
        cur.execute("ALTER TABLE document_cache ADD COLUMN extracted_json TEXT")

    conn.commit()


def init_db():
    if not CACHE_ENABLED:
        return
    conn = get_conn()
    _ensure_schema(conn)
    conn.close()


def get_document_cache(file_hash: str, doc_type: str) -> Optional[Dict[str, Any]]:
    if not CACHE_ENABLED:
        return None

    conn = get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()

    cur.execute("""
        SELECT extracted_json, created_at
        FROM document_cache
        WHERE file_hash=? AND doc_type=?
        ORDER BY id DESC
        LIMIT 1
    """, (file_hash, doc_type))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    extracted_json, created_at = row
    if not extracted_json:
        return None

    if TTL_DAYS and _is_expired(created_at):
        return None

    try:
        return json.loads(extracted_json)
    except Exception:
        return None


def save_document_cache(file_hash: str, doc_type: str, filename: str, extracted: Dict[str, Any]):
    if not CACHE_ENABLED:
        return

    conn = get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO document_cache(file_hash, doc_type, filename, extracted_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (file_hash, doc_type, filename, json.dumps(extracted, ensure_ascii=False), _now_iso()))

    conn.commit()
    conn.close()


def delete_document_cache(file_hash: str, doc_type: Optional[str] = None):
    if not CACHE_ENABLED:
        return

    conn = get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()

    if doc_type:
        cur.execute("DELETE FROM document_cache WHERE file_hash=? AND doc_type=?", (file_hash, doc_type))
    else:
        cur.execute("DELETE FROM document_cache WHERE file_hash=?", (file_hash,))

    conn.commit()
    conn.close()


def get_workflow_cache(workflow_hash: str) -> Optional[Dict[str, Any]]:
    if not CACHE_ENABLED:
        return None

    conn = get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()

    cur.execute("""
        SELECT report_json, created_at
        FROM workflow_cache
        WHERE workflow_hash=?
        LIMIT 1
    """, (workflow_hash,))

    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    report_json, created_at = row
    if TTL_DAYS and _is_expired(created_at):
        conn.close()
        return None

    # update last_access
    cur.execute("""
        UPDATE workflow_cache SET last_access=?
        WHERE workflow_hash=?
    """, (_now_iso(), workflow_hash))
    conn.commit()
    conn.close()

    try:
        return json.loads(report_json)
    except Exception:
        return None


def save_workflow_cache(workflow_hash: str, bl_hash: str, invoice_hash: str, packing_hash: Optional[str], report: Dict[str, Any]):
    if not CACHE_ENABLED:
        return

    conn = get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()

    now = _now_iso()
    cur.execute("""
        INSERT OR REPLACE INTO workflow_cache(workflow_hash, bl_hash, invoice_hash, packing_hash, report_json, created_at, last_access)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (workflow_hash, bl_hash, invoice_hash, packing_hash, json.dumps(report, ensure_ascii=False), now, now))

    conn.commit()
    conn.close()


def delete_workflow_cache(workflow_hash: str):
    if not CACHE_ENABLED:
        return

    conn = get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM workflow_cache WHERE workflow_hash=?", (workflow_hash,))
    conn.commit()
    conn.close()
    
def delete_document_cache_by_hash(file_hash: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM document_cache
        WHERE file_hash = ?
    """, (file_hash,))

    conn.commit()
    conn.close()

