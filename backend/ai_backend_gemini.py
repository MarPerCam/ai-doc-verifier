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
    ncm_4d: Optional[str] = None
    ncm_8d: Optional[str] = None
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
        
        prompt = """THIS IS A SHIPPING DOCUMENT ANALYSIS TASK.

YOU ARE ACTING AS A SENIOR BRAZILIAN CUSTOMS BROKER, TRADE COMPLIANCE ANALYST, AND SHIPPING DOCUMENT AUDITOR.

YOUR JOB IS TO EXTRACT DATA ONLY FROM THE DOCUMENT PROVIDED.
THE DOCUMENT MAY BE:
- BILL OF LADING (BL)
- PACKING LIST (PL)
- COMMERCIAL INVOICE (CI)

STRICT EXTRACTION RULES:

GENERAL BEHAVIOR:
- NEVER guess.
- NEVER infer from business logic.
- NEVER copy values from another document type.
- NEVER create missing values.
- NEVER "complete" truncated text unless the missing characters are physically visible elsewhere in the SAME document.
- ONLY use text physically visible in THIS document.
- If a value is not visible in THIS document, return null.
- This is a deterministic extraction task, not a reasoning task.
- Behave like a Receita Federal auditor validating shipping documents.

MULTI-PAGE RULES (CRITICAL):
- YOU MUST READ ALL PAGES before producing the final JSON.
- DO NOT stop after page 1.
- If the document has 2, 3, or more pages, inspect every page.
- Search for values across:
  1. page headers
  2. page footers
  3. main shipment/cargo tables
  4. continuation tables on later pages
  5. description of goods blocks
  6. right-side freight panels
  7. lower-half freight summaries
  8. party blocks (shipper / consignee)
- If the cargo table continues across pages, treat it as ONE continuous table.
- If the same total is repeated on multiple pages, DO NOT sum duplicates.
- Prefer the final total or the main shipment total when clearly labeled.
- If a field appears only on page 2+ and not on page 1, you MUST still extract it.

DOCUMENT INDEPENDENCE:
- Each document is independent.
- BL must be extracted only from BL.
- PL must be extracted only from PL.
- CI must be extracted only from CI.
- Never use Invoice data to fill BL fields.
- Never use BL data to fill Packing List fields.
- Never use Packing List data to fill missing CI fields.

PRIORITY OF LOCATIONS WITHIN THE SAME DOCUMENT:
If the same field appears more than once in the SAME document, use this priority:
1. MAIN shipment / cargo / goods table
2. Description of Goods / Cargo Details block
3. Freight summary / totals box
4. Party block (Shipper / Consignee)
5. Header / footer repetition

IGNORE:
- booking references
- invoice internal item codes
- container numbers
- seal numbers
- SKU
- model/reference numbers unless explicitly labeled as HS / NCM / Commodity Code
- notify party for consignee extraction
- any values that are not clearly associated with the requested field

====================================================
FIELDS TO EXTRACT
====================================================

RETURN ALL FIELDS ALWAYS:

{
  "shipper_name": "...",
  "consignee": "...",
  "cnpj": "...",
  "localization": "...",
  "ncm_4d": "...",
  "ncm_8d": "...",
  "packages": number,
  "gross_weight": number,
  "cbm": number
}

If a field does not physically exist, return null.

====================================================
SHIPPER / CONSIGNEE RULES
====================================================

SHIPPER:
- Extract only the SHIPPER / EXPORTER / CONSIGNOR legal name.
- Disconsideer this signals "." "," "/" "-" "_" 

CONSIGNEE:
- Extract only the CONSIGNEE legal name.
- IGNORE Notify Party completely.
- Disconsideer this signals "." "," "/" "-" "_" 
- If consignee names are similar across the document (more than 80% similarity), treat them as a match and return ONE canonical normalized value.

NAME NORMALIZATION:
- Convert to lowercase, then capitalize the first letter of each word.
- Remove dots and commas.
- Remove accents/diacritics.
- Remove duplicated spaces.
- Remove symbols like: -, _, /, \ ONLY when they are punctuation noise.
- Keep the legal name faithful to the document.
- DO NOT invent missing words.

NAME RECONCILIATION ACROSS PAGES / BLOCKS:
- If the same shipper or consignee appears multiple times in the SAME document with minor formatting differences, OCR noise, punctuation variation, or spacing variation, treat them as the SAME entity.
- If two versions are clearly the same company, return ONE canonical normalized value.
- Prefer the clearest and most complete version physically visible in the SAME document.
- If two names are only "similar" but not clearly the same legal entity, DO NOT merge them.
- Only unify them when the equivalence is obvious from the document itself.

EXAMPLES OF ACCEPTABLE SAME-ENTITY NORMALIZATION:
- "Acme Ltda." and "ACME LTDA"
- "Global Trade Importacao Ltda" and "Global Trade Importação Ltda."
- "Blue Ocean Coml. Ltda" and "Blue Ocean Coml Ltda"

EXAMPLES OF NOT SAFE TO MERGE:
- "Acme Brasil Ltda" and "Acme Trading Ltda"
- "Global Foods" and "Global Food Imports"

====================================================
CNPJ RULES
====================================================

- Extract only if a Brazilian CNPJ is physically visible in THIS document.
- CNPJ must have exactly 14 digits after cleanup.
- Accept formatted versions like:
  XX.XXX.XXX/XXXX-XX
- Remove punctuation and return digits only.
- Ignore CPF.
- If no valid 14-digit CNPJ is visible, return null.

====================================================
LOCALIZATION RULES
====================================================

- Extract only localization physically visible in THIS document.
- Prefer: City + State.
- Use the location associated with the relevant party block or official address block.
- If both city and state are clearly visible, return as:
  "City, State"
- If only city is visible and state is missing, return null.
- If only country is visible, return null.
- Never infer state from city.
- Never infer location from CNPJ.

====================================================
NCM / HS CODE RULES (EXTREMELY CRITICAL)
====================================================

The document may contain:
- HS
- HS CODE
- NCM
- Commodity Code
- Harmonized Code
- Tariff Code

You MUST search across ALL pages for these labels.

VALID LOCATIONS FOR NCM/HS:
- Description of Goods
- Cargo Details
- Product / Item table
- HS CODE column
- NCM column
- Commodity Code field
- Goods description line next to NCM/HS

REJECT AS NCM:
- SKU
- Item number
- Internal reference
- Container number
- Booking number
- Seal number
- Purchase order number
- Any code with less than 4 digits after cleanup

MULTIPLE NCM RULES (CRITICAL):
- If the document contains MORE THAN ONE valid NCM/HS code, YOU MUST RETURN ALL OF THEM.
- Keep the order of appearance from the document.
- Do NOT collapse different NCMs into one.
- Do NOT keep duplicates more than once if they are exact duplicates.
- Return multiple values separated by "/" only.
- Example:
  "ncm_4d": "8438/3923/8501"
  "ncm_8d": "84381000/39232190/85011010"

NORMALIZATION LOGIC:
1. Remove dots, spaces, hyphens, and punctuation noise.
2. Keep digits only.
3. A valid code must have AT LEAST 4 digits.

FOR ncm_4d:
- If code has 4 or more digits, take the FIRST 4 digits.
- Return all distinct valid 4-digit headings in document order, separated by "/".
- If no valid code exists, return null.

FOR ncm_8d:
- If code has 8 or more digits, take the FIRST 8 digits.
- Return all distinct valid 8-digit codes in document order, separated by "/".
- If the document shows only 4, 5, 6, or 7 digits and never shows 8 digits, return null for ncm_8d.
- NEVER invent the missing 8-digit suffix.

EXAMPLES:
- 84381000 -> ncm_4d = 8438 ; ncm_8d = 84381000
- 84.38.10.00 -> ncm_4d = 8438 ; ncm_8d = 84381000
- 8438 -> ncm_4d = 8438 ; ncm_8d = null
- 84.38.10 -> digits become 843810 -> ncm_4d = 8438 ; ncm_8d = null
- If codes are 84381000 and 39232190 -> ncm_4d = 8438/3923 ; ncm_8d = 84381000/39232190

IF MULTIPLE CODES APPEAR IN DIFFERENT PAGES:
- Read all pages first.
- Combine all valid distinct codes in order of first appearance.

====================================================
PACKAGES RULES
====================================================

Look for:
- Packages
- Pkgs
- Cartons
- Boxes
- Volumes
- Units

RULES:
- Return the TOTAL quantity physically stated in the document.
- Prefer the total from the main shipment table or total line.
- If line-item package quantities are repeated on continuation pages, do not duplicate totals.
- If only item-level counts exist and a total is not explicitly provided, sum ONLY if the rows are clearly distinct and not duplicate carryovers from another page.
- If not physically determinable with confidence, return null.

====================================================
WEIGHT RULES
====================================================

Extract GROSS WEIGHT ONLY.

Look for:
- Gross Weight
- G.W.
- Gross Wt
- Total Gross Weight

RULES:
- Normalize to kilograms.
- If weight is already in KG/KGS, return numeric value only.
- If another unit is used and a reliable conversion is not explicitly possible, return null.
- Never use Net Weight instead of Gross Weight.
- If both gross and net appear, use gross only.
- If gross weight appears multiple times, prefer the main total.

====================================================
CBM / VOLUME RULES
====================================================

Look for:
- Measurement
- CBM
- Volume
- M3
- Cubic Meters

RULES:
- Normalize to cubic meters.
- If document already shows CBM / M3, return numeric value only.
- If volume appears in ft³ and conversion is required, convert using:
  1 m³ = 35.315 ft³
- Do not round unless unavoidable.
- If multiple identical totals repeat across pages, do not sum duplicates.
- If no physically reliable total exists, return null.

====================================================
CROSS-CHECK / ANTI-HALLUCINATION RULES
====================================================

Before returning JSON, perform these checks internally:
- Did I inspect ALL pages?
- Did I extract only from THIS document?
- Did I avoid using data from BL/PL/CI interchangeably?
- Did I avoid Notify Party when extracting consignee?
- Did I keep shipper and consignee faithful to the document?
- If shipper/consignee had small formatting differences, did I normalize them conservatively?
- Did I return ALL visible valid NCMs, not just the first one?
- Did I avoid inventing an 8-digit NCM when only 4 digits are visible?
- Did I avoid copying repeated totals from multiple pages?
- Is every non-null field physically visible in the document?

If any answer is "no", correct it before returning JSON.

====================================================
FINAL OUTPUT RULES
====================================================

Return ONLY valid JSON.
NO markdown.
NO comments.
NO explanation.
NO extra keys.
NO confidence score.
NO notes.

OUTPUT EXACTLY:

{
  "shipper_name": "...",
  "consignee": "...",
  "cnpj": "...",
  "localization": "...",
  "ncm_4d": "...",
  "ncm_8d": "...",
  "packages": number,
  "gross_weight": number,
  "cbm": number
}
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
                    ncm_4d=data.get('ncm_4d'),
                    ncm_8d=data.get('ncm_8d'),
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
        
        prompt = f"""THIS IS A SHIPPING DOCUMENT ANALYSIS TASK.

YOU ARE ACTING AS A SENIOR BRAZILIAN CUSTOMS BROKER, TRADE COMPLIANCE ANALYST, AND SHIPPING DOCUMENT AUDITOR.

YOUR JOB IS TO EXTRACT DATA ONLY FROM THE DOCUMENT PROVIDED.
THE DOCUMENT MAY BE:
- BILL OF LADING (BL)
- PACKING LIST (PL)
- COMMERCIAL INVOICE (CI)

STRICT EXTRACTION RULES:

GENERAL BEHAVIOR:
- NEVER guess.
- NEVER infer from business logic.
- NEVER copy values from another document type.
- NEVER create missing values.
- NEVER “complete” truncated text unless the missing characters are physically visible elsewhere in the SAME document.
- ONLY use text physically visible in THIS document.
- If a value is not visible in THIS document, return null.
- This is a deterministic extraction task, not a reasoning task.
- Behave like a Receita Federal auditor validating shipping documents.

MULTI-PAGE RULES (CRITICAL):
- YOU MUST READ ALL PAGES before producing the final JSON.
- DO NOT stop after page 1.
- If the document has 2, 3, or more pages, inspect every page.
- If the cargo table continues across pages, treat it as ONE continuous table.
- If the same total is repeated on multiple pages, DO NOT sum duplicates.
- Prefer the final total or the main shipment total when clearly labeled.
- If a field appears only on page 2+ and not on page 1, you MUST still extract it.

DOCUMENT INDEPENDENCE:
- Each document is independent.
- BL must be extracted only from BL.
- PL must be extracted only from PL.
- CI must be extracted only from CI.
- Never use Invoice data to fill BL fields.
- Never use BL data to fill Packing List fields.
- Never use Packing List data to fill missing CI fields.

PRIORITY OF LOCATIONS WITHIN THE SAME DOCUMENT:
If the same field appears more than once in the SAME document, use this priority:
1. MAIN shipment / cargo / goods table
2. Description of Goods / Cargo Details block
3. Freight summary / totals box
4. Party block (Shipper / Consignee)
5. Header / footer repetition

IGNORE:
- booking references
- invoice internal item codes
- container numbers
- seal numbers
- SKU
- model/reference numbers unless explicitly labeled as HS / NCM / Commodity Code
- notify party for consignee extraction
- any values that are not clearly associated with the requested field

====================================================
FIELDS TO EXTRACT
====================================================

RETURN ALL FIELDS ALWAYS:

{{
  "shipper_name": "...",
  "consignee": "...",
  "cnpj": "...",
  "localization": "...",
  "ncm_4d": "...",
  "ncm_8d": "...",
  "packages": number,
  "gross_weight": number,
  "cbm": number
}}

If a field does not physically exist, return null.

====================================================
SHIPPER / CONSIGNEE RULES
====================================================

SHIPPER:
- Extract only the SHIPPER / EXPORTER / CONSIGNOR legal name.
- Disconsideer this signals "." "," "/" "-" "_" 

CONSIGNEE:
- Extract only the CONSIGNEE legal name.
- IGNORE Notify Party completely.
- Disconsideer this signals "." "," "/" "-" "_" 
- If consignee names are similar across the document (more than 80% similarity),treat them as a match and return all them equal.

NAME NORMALIZATION:
- Convert to lowercase, then capitalize the first letter of each word.
- Remove dots and commas.
- Remove accents/diacritics.
- Remove duplicated spaces.
- Remove symbols like: -, _, /, \\ ONLY when they are punctuation noise.
- Keep the legal name faithful to the document.
- DO NOT invent missing words.

NAME RECONCILIATION ACROSS PAGES / BLOCKS:
- If the same shipper or consignee appears multiple times in the SAME document with minor formatting differences, OCR noise, punctuation variation, or spacing variation, treat them as the SAME entity.
- If two versions are clearly the same company, return ONE canonical normalized value.
- Prefer the clearest and most complete version physically visible in the SAME document.
- If two names are only “similar” but not clearly the same legal entity, DO NOT merge them.
- Only unify them when the equivalence is obvious from the document itself.

EXAMPLES OF ACCEPTABLE SAME-ENTITY NORMALIZATION:
- “Acme Ltda.” and “ACME LTDA”
- “Global Trade Importacao Ltda” and “Global Trade Importação Ltda.”
- “Blue Ocean Coml. Ltda” and “Blue Ocean Coml Ltda”

EXAMPLES OF NOT SAFE TO MERGE:
- “Acme Brasil Ltda” and “Acme Trading Ltda”
- “Global Foods” and “Global Food Imports”

====================================================
CNPJ RULES
====================================================

- Extract only if a Brazilian CNPJ is physically visible in THIS document.
- CNPJ must have exactly 14 digits after cleanup.
- Accept formatted versions like:
  XX.XXX.XXX/XXXX-XX
- Remove punctuation and return digits only.
- Ignore CPF.
- If no valid 14-digit CNPJ is visible, return null.

====================================================
LOCALIZATION RULES
====================================================

- Extract only localization physically visible in THIS document.
- Prefer: City + State.
- Use the location associated with the relevant party block or official address block.
- If both city and state are clearly visible, return as:
  "City, State"
- If only city is visible and state is missing, return null.
- If only country is visible, return null.
- Never infer state from city.
- Never infer location from CNPJ.

====================================================
NCM / HS CODE RULES (EXTREMELY CRITICAL)
====================================================

The document may contain:
- HS
- HS CODE
- NCM
- Commodity Code
- Harmonized Code
- Tariff Code

You MUST search across ALL pages for these labels.

VALID LOCATIONS FOR NCM/HS:
- Description of Goods
- Cargo Details
- Product / Item table
- HS CODE column
- NCM column
- Commodity Code field
- Goods description line next to NCM/HS

REJECT AS NCM:
- SKU
- Item number
- Internal reference
- Container number
- Booking number
- Seal number
- Purchase order number
- Any code with less than 4 digits after cleanup

MULTIPLE NCM RULES (CRITICAL):
- If the document contains MORE THAN ONE valid NCM/HS code, YOU MUST RETURN ALL OF THEM.
- Keep the order of appearance from the document.
- Do NOT collapse different NCMs into one.
- Do NOT keep duplicates more than once if they are exact duplicates.
- Return multiple values separated by "/" only.
- Example:
  "ncm_4d": "8438/3923/8501"
  "ncm_8d": "84381000/39232190/85011010"

NORMALIZATION LOGIC:
1. Remove dots, spaces, hyphens, and punctuation noise.
2. Keep digits only.
3. A valid code must have AT LEAST 4 digits.

FOR ncm_4d:
- If code has 4 or more digits, take the FIRST 4 digits.
- Return all distinct valid 4-digit headings in document order, separated by "/".
- If no valid code exists, return null.

FOR ncm_8d:
- If code has 8 or more digits, take the FIRST 8 digits.
- Return all distinct valid 8-digit codes in document order, separated by "/".
- If the document shows only 4, 5, 6, or 7 digits and never shows 8 digits, return null for ncm_8d.
- NEVER invent the missing 8-digit suffix.

EXAMPLES:
- 84381000 -> ncm_4d = 8438 ; ncm_8d = 84381000
- 84.38.10.00 -> ncm_4d = 8438 ; ncm_8d = 84381000
- 8438 -> ncm_4d = 8438 ; ncm_8d = null
- 84.38.10 -> digits become 843810 -> ncm_4d = 8438 ; ncm_8d = null
- If codes are 84381000 and 39232190 -> ncm_4d = 8438/3923 ; ncm_8d = 84381000/39232190

IF MULTIPLE CODES APPEAR IN DIFFERENT PAGES:
- Read all pages first.
- Combine all valid distinct codes in order of first appearance.

====================================================
PACKAGES RULES
====================================================

Look for:
- Packages
- Pkgs
- Cartons
- Boxes
- Volumes
- Units

RULES:
- Return the TOTAL quantity physically stated in the document.
- Prefer the total from the main shipment table or total line.
- If line-item package quantities are repeated on continuation pages, do not duplicate totals.
- If only item-level counts exist and a total is not explicitly provided, sum ONLY if the rows are clearly distinct and not duplicate carryovers from another page.
- If not physically determinable with confidence, return null.

====================================================
WEIGHT RULES
====================================================

Extract GROSS WEIGHT ONLY.

Look for:
- Gross Weight
- G.W.
- Gross Wt
- Total Gross Weight

RULES:
- Normalize to kilograms.
- If weight is already in KG/KGS, return numeric value only.
- If another unit is used and a reliable conversion is not explicitly possible, return null.
- Never use Net Weight instead of Gross Weight.
- If both gross and net appear, use gross only.
- If gross weight appears multiple times, prefer the main total.

====================================================
CBM / VOLUME RULES
====================================================

Look for:
- Measurement
- CBM
- Volume
- M3
- Cubic Meters

RULES:
- Normalize to cubic meters.
- If document already shows CBM / M3, return numeric value only.
- If volume appears in ft³ and conversion is required, convert using:
  1 m³ = 35.315 ft³
- Do not round unless unavoidable.
- If multiple identical totals repeat across pages, do not sum duplicates.
- If no physically reliable total exists, return null.

====================================================
CROSS-CHECK / ANTI-HALLUCINATION RULES
====================================================

Before returning JSON, perform these checks internally:
- Did I inspect ALL pages?
- Did I extract only from THIS document?
- Did I avoid using data from BL/PL/CI interchangeably?
- Did I avoid Notify Party when extracting consignee?
- Did I keep shipper and consignee faithful to the document?
- If shipper/consignee had small formatting differences, did I normalize them conservatively?
- Did I return ALL visible valid NCMs, not just the first one?
- Did I avoid inventing an 8-digit NCM when only 4 digits are visible?
- Did I avoid copying repeated totals from multiple pages?
- Is every non-null field physically visible in the document?

If any answer is "no", correct it before returning JSON.

====================================================
FINAL OUTPUT RULES
====================================================

Return ONLY valid JSON.
NO markdown.
NO comments.
NO explanation.
NO extra keys.
NO confidence score.
NO notes.

OUTPUT EXACTLY:

{{
  "shipper_name": "...",
  "consignee": "...",
  "cnpj": "...",
  "localization": "...",
  "ncm_4d": "...",
  "ncm_8d": "...",
  "packages": number,
  "gross_weight": number,
  "cbm": number
}}

TEXT CONTENT TO ANALYZE:
========================
{text}
"""

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
                    ncm_4d=data.get('ncm_4d'),
                    ncm_8d=data.get('ncm_8d'),
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
                "field": "NCM 4 Digits",
                "key": "ncm_4d",
                "docs": [("BL", bl), ("Invoice", invoice)],
                "type": "text"
            },
            {
                "field": "NCM 8 Digits",
                "key": "ncm_8d",
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
            if comp["key"] in ("ncm_4d", "ncm_8d"):
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
