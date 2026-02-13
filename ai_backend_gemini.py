"""
AI-Powered Document Verification System - Backend
Real document extraction using Google Gemini API with Vision
"""

import os
import re
import json
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from pathlib import Path
import PyPDF2
import openpyxl
from PIL import Image
import io

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


@dataclass
class DocumentData:
    """Structure for extracted shipping document data"""
    shipper_name: Optional[str] = None
    consignee: Optional[str] = None
    cnpj: Optional[str] = None
    localization: Optional[str] = None
    ncm: Optional[str] = None
    packages: Optional[int] = None
    gross_weight: Optional[float] = None
    cbm: Optional[float] = None
    raw_text: Optional[str] = None
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None


class AIDocumentExtractor:

    def __init__(self, api_key: str = None):
        # priority:
        # 1. parametro
        # 2. variavel ambiente
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Please provide API key.")

    def extract_from_file(self, file_path: str, doc_type: str = "unknown") -> DocumentData:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext == '.pdf':
            result = self._extract_from_pdf(file_path, doc_type)
        elif file_ext in ['.xlsx', '.xls']:
            result = self._extract_from_excel(file_path, doc_type)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            result = self._extract_from_image(file_path, doc_type)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Safety fallback
        if result is None:
            return DocumentData(extraction_method="Extraction returned None", confidence=0.0)

        return result


    def _extract_from_pdf(self, file_path: Path, doc_type: str) -> DocumentData:
        extracted_text = ""

        try:
            reader = PyPDF2.PdfReader(str(file_path))
            texts = []
            for page in reader.pages[:2]:
                t = page.extract_text()
                if t:
                    texts.append(t)

            extracted_text = "\n".join(texts).strip()
        except Exception:
            extracted_text = ""

        with open(file_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        extracted = self._call_gemini_vision(
            pdf_data,
            "application/pdf",
            doc_type,
            extracted_text
        )

        extracted.extraction_method = "Gemini Hybrid PDF"
        return extracted


    
    def _extract_from_image(self, file_path: Path, doc_type: str) -> DocumentData:
        """Extract data from image using Gemini Vision API"""
        
        # Read image as base64
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine media type
        ext = file_path.suffix.lower()
        media_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        
        # Use Gemini to extract data
        extracted = self._call_gemini_vision(
            image_data,
            media_type,
            doc_type
        )
        
        extracted.extraction_method = 'Gemini AI Vision (Image)'
        return extracted
    
    def _extract_from_excel(self, file_path: Path, doc_type: str) -> DocumentData:
        """Extract data from Excel file"""
        
        try:
            # Read Excel file
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
            
            # Convert to text representation
            text_content = []
            for row in sheet.iter_rows(values_only=True):
                row_text = ' | '.join([str(cell) if cell is not None else '' for cell in row])
                if row_text.strip():
                    text_content.append(row_text)
            
            text = '\n'.join(text_content)
            
            # Use Gemini to parse the text
            extracted = self._call_gemini_text(text, doc_type)
            extracted.extraction_method = 'Gemini AI Text (Excel)'
            extracted.raw_text = text[:500]  # Store sample
            
            return extracted
            
        except Exception as e:
            print(f"Excel extraction error: {e}")
            return DocumentData(extraction_method=f'Excel Error: {str(e)}')
    
    def _call_gemini_vision(
        self,
        base64_data: str,
        media_type: str,
        doc_type: str,
        extracted_text: str = ""
    ) -> DocumentData:
        """Call Gemini API with vision capabilities"""
        
        doc_type_descriptions = {
            'bl': 'Bill of Lading',
            'invoice': 'Commercial Invoice',
            'packing': 'Packing List'
        }
        bl_focus = ""

        if doc_type == "bl":
            bl_focus = """
THIS IS A BILL OF LADING (BL).

YOU ARE ACTING AS A SENIOR BRAZILIAN CUSTOMS BROKER AND SHIPPING DOCUMENT ANALYZER.

YOUR JOB IS TO EXTRACT DATA ONLY FROM THIS BILL OF LADING.

STRICT EXTRACTION RULES FOR BILL OF LADING:

GENERAL:

- NEVER guess values.
- NEVER infer values from other documents.
- NEVER copy values from Invoice or Packing List.
- Only extract what is physically visible in THIS BL.

SEARCH LOCATIONS (VERY IMPORTANT):

Most BLs contain data inside:

- Shipment tables
- Cargo description tables
- Boxes titled: “Cargo Details”, “Description of Goods”, “Marks & Numbers”
- Columns labeled:
  - Shipper / Exporter / Consignor
  - Consignee
  - Packages / Pkgs / Cartons / Units
  - Gross Weight / G.W.
  - Measurement / Volume / CBM / M3
  - HS Code / NCM / Commodity Code

You MUST inspect:

1. Main cargo table  
2. Description of goods block  
3. Any boxed freight summary  
4. Lower half of the document  
5. Right-hand freight panels  

If the same value appears multiple times, ALWAYS use the MAIN shipment table.

---

SHIPPER & CONSIGNEE:

- Extract ONLY the CONSIGNEE (ignore Notify Party completely).
- Normalize names:
  lowercase everything, then capitalize first letter of each word.
- Remove dots and commas.

---

NCM / HS CODE RULES (CRITICAL):

Brazilian NCM format:

- EXACTLY 8 numeric digits
- Example: 84381000
- May appear formatted as:
  84.38.10.00
  8438.10.00
  84381000

YOU MUST normalize to 8 digits only.

ACCEPT ONLY IF:

- Contains exactly 8 digits after cleaning.
- Appears next to labels:
  "NCM"
  "HS CODE"
  "HS"
  "Commodity Code"
  "Harmonized Code"

IGNORE:

- 6 digit HS
- internal product codes
- invoice item codes
- container numbers
- booking references

COMMON BL LOCATIONS FOR NCM:

- Inside Description of Goods
- Inside Cargo table
- Under HS CODE column
- Near commodity description

If multiple codes exist:

- choose the FIRST valid 8 digit NCM.

If NO valid 8-digit NCM exists:

RETURN null.

DO NOT invent NCM.

---

PACKAGES:

Look for keywords:

Packages  
Pkgs  
Cartons  
Boxes  
Volumes  

Return TOTAL quantity.

---

WEIGHT:

Extract Gross Weight ONLY.

Normalize to kilograms.

---

CBM / VOLUME:

Extract Measurement / CBM / Volume.

Normalize to cubic meters.

---

FINAL RULE:

If ANY field does NOT physically exist inside this BL:

RETURN null.

OUTPUT FORMAT:

Return ONLY JSON:

{
  "shipper_name": "...",
  "consignee": "...",
  "cnpj": "...",
  "localization": "...",
  "ncm": "...",
  "packages": number,
  "gross_weight": number,
  "cbm": number
}

NO explanations.
NO markdown.
NO comments.
ONLY JSON.

        """

        doc_description = doc_type_descriptions.get(doc_type, 'shipping document')
        
        prompt = bl_focus + """
You are an expert document analyzer for international shipping. 
Analyze this {doc_description} and extract the following information with high precision:
You are a SENIOR BRAZILIAN CUSTOMS BROKER and INTERNATIONAL SHIPPING DOCUMENT ANALYZER.

Model: Gemini 2.5 Flash (Vision + Text).

Your task is deterministic structured extraction.

NO creativity.
NO guessing.
NO assumptions.

You MUST behave like Receita Federal auditing documents.

---

REQUIRED FIELDS:

1. Shipper Name
2. Consignee Name
3. CNPJ (14 digits)
4. Localization (City + State)
5. NCM (8 digits Brazilian classification)
6. Packages
7. Gross Weight (kg)
8. CBM (m³)

---

DOCUMENT TYPES:

Bill of Lading (BL)
Commercial Invoice (CI)
Packing List (PL)

Each document is independent.

Never copy values between documents.

---

SHIPPER / CONSIGNEE:

- Extract legal company names.
- Normalize:
  lowercase → capitalize first letter per word.
- Remove dots and commas.
- Remove "acentos" (á, é, í, ó, ú, â, ê, î, ô, û, ã, õ, ç)
- Remove - _ / \ symbols
- Disconsider " " , don't remove only Disconsider

---

CNPJ RULES:

Brazilian CNPJ:

- Exactly 14 digits
- May appear formatted:
  XX.XXX.XXX/XXXX-XX

Accept only if 14 digits after cleanup.

Ignore CPF.

---

NCM RULES (EXTREMELY IMPORTANT):

Brazilian NCM:

- ALWAYS 4 numeric digits
- Examples:
  8438
  84.38
  8438

Normalize to:

8438

VALID ONLY IF:

- After cleanup contains exactly 4 digits
- Appears near:

"NCM"
"HS CODE"
"HS"
"Commodity Code"
"Harmonized"

REJECT:

- 6 digit HS
- SKU
- Item numbers
- Product IDs
- container numbers

DOCUMENT SOURCES:

BL:
- Description of Goods
- Cargo table
- HS CODE column

Invoice:
- Item table
- Product description
- Fiscal classification

Packing:
- Rarely contains NCM
- Only accept if explicitly labeled

PRIORITY ORDER:

1️⃣ BL  
2️⃣ Invoice  
3️⃣ Packing (last resort)

If missing in BL → null for BL.

If missing in BOTH BL and Invoice → null.

Never infer.

---

PACKAGES:

Extract TOTAL quantity using:

Don't create values, only extract what exists in the document.

packages  
cartons  
boxes  
volumes  
pkgs  

---

WEIGHT:

Extract GROSS weight only.

Convert to KG if needed.

---

CBM:

Extract Measurement / Volume.

Convert ft³ → m³  
(1 m³ = 35.315 ft³)

---

NUMERIC PRECISION:

Never round unless unavoidable.

---

NULL RULE:

If field does not physically exist → null.

---

JSON OUTPUT:

Return ALL fields always:

{{
  "shipper_name": "...",
  "consignee": "...",
  "cnpj": "...",
  "localization": "City, State",
  "ncm": "...",
  "packages": number,
  "gross_weight": number,
  "cbm": number
}}

NO additional text.
NO explanation.
ONLY JSON.

Failure to follow this schema is considered an extraction error.
"""

        try:
            # Prepare Gemini API request
            url = f"{GEMINI_API_URL}?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Gemini API format
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": base64_data
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API Error {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract text from Gemini response
            text_content = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            text_content += part['text']
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text_content)
            if json_match:
                data = json.loads(json_match.group())
                return DocumentData(
                    shipper_name=data.get('shipper_name'),
                    consignee=data.get('consignee'),
                    cnpj=data.get('cnpj'),
                    localization=data.get('localization'),
                    ncm=data.get('ncm'),
                    packages=int(data.get('packages')) if data.get('packages') else None,
                    gross_weight=float(data.get('gross_weight')) if data.get('gross_weight') else None,
                    cbm=float(data.get('cbm')) if data.get('cbm') else None,
                    raw_text=text_content[:200],
                    confidence=0.9
                )
            else:
                raise Exception("Could not parse JSON from Gemini response")
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return DocumentData(
                extraction_method=f'Gemini AI Error: {str(e)}',
                confidence=0.0
            )
    
    def _call_gemini_text(self, text: str, doc_type: str) -> DocumentData:
        """Call Gemini API with text content (for Excel)"""
        
        prompt = f"""Extract shipping document information from this text.

Text content:
{text}

Extract these fields and return as JSON:
{{
    "shipper_name": "...",
    "consignee": "...",
    "cnpj": "...",
    "localization": "...",
    "ncm": "...",
    "packages": number,
    "gross_weight": number,
    "cbm": number
}}

Return only the JSON, no explanation."""

        try:
            url = f"{GEMINI_API_URL}?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API Error {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract text
            text_content = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            text_content += part['text']
            
            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', text_content)
            if json_match:
                data = json.loads(json_match.group())
                return DocumentData(
                    shipper_name=data.get('shipper_name'),
                    consignee=data.get('consignee'),
                    cnpj=data.get('cnpj'),
                    localization=data.get('localization'),
                    ncm=data.get('ncm'),
                    packages=int(data.get('packages')) if data.get('packages') else None,
                    gross_weight=float(data.get('gross_weight')) if data.get('gross_weight') else None,
                    cbm=float(data.get('cbm')) if data.get('cbm') else None,
                    confidence=0.85
                )
            
            raise Exception("Could not parse JSON response")
            
        except Exception as e:
            print(f"Gemini text API error: {e}")
            return DocumentData(extraction_method=f'Error: {str(e)}')


class CNPJValidator:
    """Validate CNPJ with format check and online validation"""
    
    @staticmethod
    def format_cnpj(cnpj: str) -> str:
        """Format CNPJ with dots, slash and hyphen"""
        clean = re.sub(r'\D', '', cnpj)
        if len(clean) == 14:
            return f"{clean[:2]}.{clean[2:5]}.{clean[5:8]}/{clean[8:12]}-{clean[12:]}"
        return cnpj
    
    @staticmethod
    def validate_format(cnpj: str) -> bool:
        """Validate CNPJ format and check digits"""
        cnpj = re.sub(r'\D', '', cnpj)
        
        if len(cnpj) != 14:
            return False
        
        if cnpj == cnpj[0] * 14:
            return False
        
        # Validate check digits
        def calc_digit(cnpj_part: str, weights: List[int]) -> int:
            total = sum(int(d) * w for d, w in zip(cnpj_part, weights))
            remainder = total % 11
            return 0 if remainder < 2 else 11 - remainder
        
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        
        digit1 = calc_digit(cnpj[:12], weights1)
        digit2 = calc_digit(cnpj[:13], weights2)
        
        return cnpj[-2:] == f"{digit1}{digit2}"
    
    @staticmethod
    def validate_online(cnpj: str) -> Dict[str, Any]:
        """Validate CNPJ using ReceitaWS API"""
        cnpj_clean = re.sub(r'\D', '', cnpj)
        
        if not CNPJValidator.validate_format(cnpj_clean):
            return {"valid": False, "error": "Invalid CNPJ format"}
        
        try:
            url = f"https://www.receitaws.com.br/v1/cnpj/{cnpj_clean}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'ERROR':
                    return {"valid": False, "error": data.get('message')}
                
                return {
                    "valid": True,
                    "cnpj": CNPJValidator.format_cnpj(cnpj_clean),
                    "razao_social": data.get('nome'),
                    "nome_fantasia": data.get('fantasia'),
                    "situacao": data.get('situacao'),
                    "endereco": f"{data.get('logradouro', '')}, {data.get('numero', '')}",
                    "cidade": data.get('municipio'),
                    "estado": data.get('uf'),
                    "cep": data.get('cep')
                }
            
            return {"valid": False, "error": f"API returned status {response.status_code}"}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}


class DocumentComparator:
    """Compare extracted data from multiple documents"""
    
    @staticmethod
    def compare(
        bl: DocumentData, 
        invoice: DocumentData, 
        packing: DocumentData = None
    ) -> Dict[str, Any]:
        """
        Compare documents and return detailed results
        
        Returns:
            Dictionary with comparison results
        """
        
        results = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "details": []
        }
        
        # Define fields to compare
        comparisons = [
            {
                "field": "Shipper Name",
                "key": "shipper_name",
                "docs": [("BL", bl), ("Invoice", invoice)],
                "type": "text"
            },
            {
                "field": "Consignee",
                "key": "consignee",
                "docs": [("BL", bl), ("Invoice", invoice), ("Packing", packing)] if packing else [("BL", bl), ("Invoice", invoice)],
                "type": "text"
            },
            {
                "field": "CNPJ",
                "key": "cnpj",
                "docs": [("BL", bl), ("Invoice", invoice)],
                "type": "cnpj"
            },
            {
                "field": "NCM Code",
                "key": "ncm",
                "docs": [("BL", bl), ("Invoice", invoice)],
                "type": "text"
            },
            {
                "field": "Number of Packages",
                "key": "packages",
                "docs": [("BL", bl), ("Invoice", invoice), ("Packing", packing)] if packing else [("BL", bl), ("Invoice", invoice)],
                "type": "numeric"
            },
            {
                "field": "Gross Weight (kg)",
                "key": "gross_weight",
                "docs": [("BL", bl), ("Invoice", invoice), ("Packing", packing)] if packing else [("BL", bl), ("Invoice", invoice)],
                "type": "numeric",
                "tolerance": 0.02  # 2% tolerance
            },
            {
                "field": "CBM (m³)",
                "key": "cbm",
                "docs": [("BL", bl), ("Invoice", invoice), ("Packing", packing)] if packing else [("BL", bl), ("Invoice", invoice)],
                "type": "numeric",
                "tolerance": 0.02
            }
        ]
        
        for comp in comparisons:
            # Get values from each document
            values = {}
            for doc_name, doc in comp["docs"]:
                if doc is not None:
                    val = getattr(doc, comp["key"], None)
                    if val is not None:
                        values[doc_name] = val
            
            # Skip if no values found
            if not values:
                continue
                        # Special rule for NCM: must exist in BOTH BL and Invoice
            if comp["key"] == "ncm":
                required_docs = {"BL", "Invoice"}
                if set(values.keys()) != required_docs:
                    results["total_checks"] += 1
                    results["failed"] += 1
                    results["details"].append({
                        "field": comp["field"],
                        "values": values,
                        "status": "mismatch"
                    })
                    continue

            results["total_checks"] += 1
            
            # Compare based on type
            if comp["type"] == "text":
                match = DocumentComparator._compare_text(values)
            elif comp["type"] == "cnpj":
                match = DocumentComparator._compare_cnpj(values)
            elif comp["type"] == "numeric":
                match = DocumentComparator._compare_numeric(values, comp.get("tolerance", 0))
            else:
                match = False
            
            if match:
                results["passed"] += 1
                status = "match"
            else:
                results["failed"] += 1
                status = "mismatch"
            
            results["details"].append({
                "field": comp["field"],
                "values": values,
                "status": status
            })
        
        return results
    
    @staticmethod
    def _compare_text(values: Dict[str, str]) -> bool:
        """Compare text values (case-insensitive, whitespace normalized)"""
        normalized = [v.strip().lower() for v in values.values() if v]
        return len(set(normalized)) == 1
    
    @staticmethod
    def _compare_cnpj(values: Dict[str, str]) -> bool:
        """Compare CNPJ values (numbers only)"""
        cleaned = [re.sub(r'\D', '', str(v)) for v in values.values() if v]
        return len(set(cleaned)) == 1
    
    @staticmethod
    def _compare_numeric(values: Dict[str, float], tolerance: float = 0) -> bool:
        """Compare numeric values with optional tolerance"""
        nums = [float(v) for v in values.values() if v is not None]
        if not nums:
            return False
        
        if tolerance == 0:
            return len(set(nums)) == 1
        
        # Check if all values are within tolerance of each other
        min_val = min(nums)
        max_val = max(nums)
        avg_val = sum(nums) / len(nums)
        
        return all(abs(n - avg_val) / avg_val <= tolerance for n in nums)


def main():
    """Example usage"""
    import sys
    
    print("="*70)
    print("AI-POWERED DOCUMENT VERIFICATION SYSTEM")
    print("Using Google Gemini AI")
    print("="*70)
    print()
    
    # Check for API key
    if not GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY not set")
        print()
        print("Please set your API key:")
        print("   Windows: set GEMINI_API_KEY=your-key-here")
        print("   Linux/Mac: export GEMINI_API_KEY=your-key-here")
        print()
        print("Or add to .env file:")
        print("   GEMINI_API_KEY=your-key-here")
        print()
        print("Get your free API key at:")
        print("   https://makersuite.google.com/app/apikey")
        return
    
    print("✓ Gemini API Key configured")
    print()
    
    # Test CNPJ validation
    print("Testing CNPJ Validation:")
    print("-" * 70)
    
    test_cnpj = "11.222.333/0001-81"
    validator = CNPJValidator()
    
    is_valid = validator.validate_format(test_cnpj)
    print(f"CNPJ: {test_cnpj}")
    print(f"Format Valid: {'✓' if is_valid else '✗'}")
    print()
    
    # Example: Extract from files (if provided)
    if len(sys.argv) > 1:
        print("Extracting data from documents:")
        print("-" * 70)
        
        extractor = AIDocumentExtractor()
        
        for file_path in sys.argv[1:]:
            if os.path.exists(file_path):
                print(f"\nProcessing: {file_path}")
                try:
                    data = extractor.extract_from_file(file_path)
                    print(f"  Shipper: {data.shipper_name}")
                    print(f"  Consignee: {data.consignee}")
                    print(f"  CNPJ: {data.cnpj}")
                    print(f"  Packages: {data.packages}")
                    print(f"  Weight: {data.gross_weight} kg")
                    print(f"  CBM: {data.cbm} m³")
                except Exception as e:
                    print(f"  Error: {e}")
    else:
        print("Usage: python ai_backend.py <file1.pdf> <file2.xlsx> ...")
        print()
        print("Example:")
        print("   python ai_backend.py bill_of_lading.pdf invoice.xlsx packing.pdf")
    
    print()
    print("="*70)


if __name__ == "__main__":
    main()
