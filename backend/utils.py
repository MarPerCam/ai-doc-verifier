import hashlib
from pathlib import Path
from typing import Dict, Optional, Any


def sha256_file(file_path: str) -> str:
    """Hash SHA-256 do conteúdo do arquivo (usado para cache)."""
    p = Path(file_path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_file_hash(file_path: str) -> str:
    """Alias (para compatibilidade com seu código)."""
    return sha256_file(file_path)


def workflow_hash(bl_hash: str, invoice_hash: str, packing_hash: Optional[str] = None) -> str:
    """Hash do workflow com base nos hashes dos arquivos."""
    base = f"bl={bl_hash}|invoice={invoice_hash}|packing={packing_hash or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def calculate_workflow_hash(files: Dict[str, str]) -> str:
    """
    Recebe dict: {"bl": path, "invoice": path, "packing": path(opcional)}
    e calcula hash do workflow pelo CONTEÚDO dos arquivos.
    """
    bl = files.get("bl")
    inv = files.get("invoice")
    pk = files.get("packing")

    if not bl or not inv:
        raise ValueError("calculate_workflow_hash requires at least 'bl' and 'invoice' paths")

    bl_h = sha256_file(bl)
    inv_h = sha256_file(inv)
    pk_h = sha256_file(pk) if pk else None
    return workflow_hash(bl_h, inv_h, pk_h)


def is_meaningful_extraction(data: Dict[str, Any]) -> bool:
    """Retorna True se a extração tem pelo menos algum campo realmente útil."""
    if not isinstance(data, dict):
        return False
    keys = ["shipper_name", "consignee", "cnpj", "localization", "ncm", "packages", "gross_weight", "cbm"]
    for k in keys:
        v = data.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return True
    return False
