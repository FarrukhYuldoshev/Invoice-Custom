# import pytesseract
# from PIL import Image
# import cv2
# import numpy as np
# import re
# from typing import Dict, List, Any, Optional, Tuple
# import pandas as pd
# from dataclasses import dataclass
# from collections import defaultdict
#
#
# @dataclass
# class TableCell:
#     """Represents a cell in a table with its position and content"""
#
#     text: str
#     left: int
#     top: int
#     width: int
#     height: int
#     row_idx: int = 0
#     col_idx: int = 0
#
#
# @dataclass
# class Table:
#     """Represents a detected table structure"""
#
#     cells: List[List[TableCell]]
#     top: int
#     left: int
#     width: int
#     height: int
#
#
# class InvoiceOCRProcessor:
#     def __init__(
#         self,
#         tesseract_cmd: Optional[str] = None,
#         languages: str = "eng+rus+chi_sim",
#         table_detection_threshold: float = 0.7,
#     ):
#         """
#         Initialize the processor
#
#         Args:
#             tesseract_cmd: Path to tesseract executable if not in PATH
#             languages: Tesseract language codes
#             table_detection_threshold: Confidence threshold for table detection
#         """
#         if tesseract_cmd:
#             pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
#         self.languages = languages
#         self.table_threshold = table_detection_threshold
#
#     def preprocess_image(self, image_path: str) -> np.ndarray:
#         """
#         Enhanced preprocessing for better OCR accuracy
#         """
#         # Read image
#         img = cv2.imread(image_path)
#
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # Apply bilateral filter to reduce noise while keeping edges sharp
#         denoised = cv2.bilateralFilter(gray, 9, 75, 75)
#
#         # Apply adaptive thresholding for better results with varying lighting
#         thresh = cv2.adaptiveThreshold(
#             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#         )
#
#         # Deskew image if needed
#         thresh = self.deskew_image(thresh)
#
#         # Resize if too small
#         height, width = thresh.shape
#         if width < 1500:
#             scale = 1500 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             thresh = cv2.resize(
#                 thresh, (new_width, new_height), interpolation=cv2.INTER_CUBIC
#             )
#
#         return thresh
#
#     def deskew_image(self, image: np.ndarray) -> np.ndarray:
#         """
#         Deskew image to improve OCR accuracy
#         """
#         coords = np.column_stack(np.where(image > 0))
#         angle = cv2.minAreaRect(coords)[-1]
#
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle
#
#         if abs(angle) > 0.5:  # Only deskew if angle is significant
#             (h, w) = image.shape[:2]
#             center = (w // 2, h // 2)
#             M = cv2.getRotationMatrix2D(center, angle, 1.0)
#             rotated = cv2.warpAffine(
#                 image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
#             )
#             return rotated
#
#         return image
#
#     def detect_lines(self, image: np.ndarray) -> Tuple[List, List]:
#         """
#         Detect horizontal and vertical lines in the image for table detection
#         """
#         # Convert to binary if not already
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         else:
#             binary = image.copy()
#
#         # Detect horizontal lines
#         horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
#         horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
#
#         # Detect vertical lines
#         vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
#         vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
#
#         # Find contours
#         h_contours, _ = cv2.findContours(
#             horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         v_contours, _ = cv2.findContours(
#             vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#
#         h_lines = [cv2.boundingRect(c) for c in h_contours]
#         v_lines = [cv2.boundingRect(c) for c in v_contours]
#
#         return h_lines, v_lines
#
#     def ocr_to_data(self, image_path: str, preprocess: bool = True) -> pd.DataFrame:
#         """
#         Enhanced OCR with better configuration
#         """
#         if preprocess:
#             img = self.preprocess_image(image_path)
#         else:
#             img = cv2.imread(image_path)
#
#         # Detect language
#         lang = (
#             self.detect_language(image_path)
#             if hasattr(self, "auto_detect")
#             else self.languages
#         )
#
#         # Enhanced OCR configuration
#         custom_config = (
#             f"-l {lang} "
#             "--oem 3 "  # Use LSTM OCR Engine
#             "--psm 6 "  # Uniform block of text
#             "-c preserve_interword_spaces=1 "
#             "-c tessedit_create_hocr=1 "  # Create HOCR output for better structure
#             "-c tessedit_pageseg_mode=6"
#         )
#
#         # Get OCR data with structure
#         data = pytesseract.image_to_data(
#             img,
#             output_type=pytesseract.Output.DATAFRAME,
#             config=custom_config,
#             lang=lang,
#         )
#
#         # Clean and filter data
#         data = data[data["text"].notna() & (data["text"] != "")]
#         data["text"] = data["text"].apply(
#             lambda x: str(x).encode("utf-8", errors="ignore").decode("utf-8").strip()
#         )
#
#         # Add confidence filtering
#         data = data[data["conf"] > 30]  # Remove low confidence detections
#
#         return data
#
#     def cluster_elements(
#         self, ocr_data: pd.DataFrame, axis: str = "y", threshold: int = 10
#     ) -> Dict[int, List]:
#         """
#         Cluster OCR elements by position (for row/column detection)
#         """
#         clusters = defaultdict(list)
#
#         if axis == "y":
#             positions = ocr_data["top"].values
#         else:  # x-axis
#             positions = ocr_data["left"].values
#
#         if len(positions) == 0:
#             return clusters
#
#         # Sort positions
#         sorted_positions = sorted(enumerate(positions), key=lambda x: x[1])
#
#         # Cluster nearby positions
#         cluster_id = 0
#         clusters[cluster_id] = [sorted_positions[0][0]]
#
#         for i in range(1, len(sorted_positions)):
#             idx, pos = sorted_positions[i]
#             prev_pos = sorted_positions[i - 1][1]
#
#             if abs(pos - prev_pos) <= threshold:
#                 clusters[cluster_id].append(idx)
#             else:
#                 cluster_id += 1
#                 clusters[cluster_id] = [idx]
#
#         return clusters
#
#     def detect_tables_advanced(
#         self, ocr_data: pd.DataFrame, image: np.ndarray = None
#     ) -> List[Table]:
#         """
#         Advanced table detection using multiple heuristics
#         """
#         tables = []
#
#         if len(ocr_data) == 0:
#             return tables
#
#         # Method 1: Detect using structural lines
#         if image is not None:
#             h_lines, v_lines = self.detect_lines(image)
#             if len(h_lines) > 2 and len(v_lines) > 2:
#                 # Find table regions based on line intersections
#                 # This indicates structured table with borders
#                 pass
#
#         # Method 2: Detect using text alignment patterns
#         # Cluster by rows (y-axis)
#         row_clusters = self.cluster_elements(ocr_data, axis="y", threshold=15)
#
#         # Find potential table regions
#         table_candidates = []
#         consecutive_rows = []
#
#         for row_id in sorted(row_clusters.keys()):
#             row_indices = row_clusters[row_id]
#             row_data = ocr_data.iloc[row_indices]
#
#             # Check if row has multiple aligned columns
#             if len(row_data) >= 2:
#                 # Cluster by columns (x-axis) within this row
#                 col_clusters = self.cluster_elements(row_data, axis="x", threshold=50)
#
#                 if len(col_clusters) >= 2:  # Multiple columns detected
#                     consecutive_rows.append(
#                         {
#                             "row_id": row_id,
#                             "data": row_data,
#                             "col_count": len(col_clusters),
#                             "col_clusters": col_clusters,
#                         }
#                     )
#                 else:
#                     # Break in table structure
#                     if len(consecutive_rows) >= 2:
#                         table_candidates.append(consecutive_rows)
#                     consecutive_rows = []
#             else:
#                 # Break in table structure
#                 if len(consecutive_rows) >= 2:
#                     table_candidates.append(consecutive_rows)
#                 consecutive_rows = []
#
#         # Don't forget the last group
#         if len(consecutive_rows) >= 2:
#             table_candidates.append(consecutive_rows)
#
#         # Convert candidates to Table objects
#         for candidate in table_candidates:
#             table = self.build_table_structure(candidate, ocr_data)
#             if table:
#                 tables.append(table)
#
#         # Method 3: Detect using keywords that typically appear in tables
#         table_keywords = [
#             "quantity",
#             "qty",
#             "price",
#             "amount",
#             "total",
#             "subtotal",
#             "description",
#             "item",
#             "unit",
#             "rate",
#             "tax",
#             "discount",
#             "no.",
#             "#",
#             "product",
#             "service",
#         ]
#
#         keyword_rows = ocr_data[
#             ocr_data["text"]
#             .str.lower()
#             .str.contains("|".join(table_keywords), na=False)
#         ]
#
#         if len(keyword_rows) > 0:
#             # Find surrounding rows that might be part of the table
#             for _, keyword_row in keyword_rows.iterrows():
#                 table_region = self.extract_table_region(keyword_row, ocr_data)
#                 if table_region is not None and len(table_region) > 0:
#                     table = self.build_table_from_region(table_region)
#                     if table and table not in tables:
#                         tables.append(table)
#
#         return tables
#
#     def build_table_structure(
#         self, rows_data: List[Dict], ocr_data: pd.DataFrame
#     ) -> Optional[Table]:
#         """
#         Build a structured Table object from detected rows
#         """
#         if not rows_data:
#             return None
#
#         # Determine common column positions across all rows
#         all_x_positions = []
#         for row in rows_data:
#             for idx_list in row["col_clusters"].values():
#                 for idx in idx_list:
#                     all_x_positions.append(row["data"].iloc[idx]["left"])
#
#         if not all_x_positions:
#             return None
#
#         # Find column boundaries
#         all_x_positions = sorted(set(all_x_positions))
#         col_boundaries = self.find_column_boundaries(all_x_positions)
#
#         # Build table cells
#         cells = []
#         for row in rows_data:
#             row_cells = []
#             for col_idx, (col_start, col_end) in enumerate(col_boundaries):
#                 # Find text in this column for this row
#                 cell_text = ""
#                 for idx in range(len(row["data"])):
#                     item = row["data"].iloc[idx]
#                     if col_start <= item["left"] <= col_end:
#                         if cell_text:
#                             cell_text += " "
#                         cell_text += str(item["text"])
#
#                 if cell_text:
#                     cell = TableCell(
#                         text=cell_text,
#                         left=col_start,
#                         top=row["data"].iloc[0]["top"],
#                         width=col_end - col_start,
#                         height=row["data"].iloc[0]["height"],
#                         row_idx=len(cells),
#                         col_idx=col_idx,
#                     )
#                     row_cells.append(cell)
#
#             if row_cells:
#                 cells.append(row_cells)
#
#         if not cells:
#             return None
#
#         # Calculate table bounds
#         min_left = min(cell.left for row in cells for cell in row)
#         min_top = min(cell.top for row in cells for cell in row)
#         max_right = max(cell.left + cell.width for row in cells for cell in row)
#         max_bottom = max(cell.top + cell.height for row in cells for cell in row)
#
#         return Table(
#             cells=cells,
#             left=min_left,
#             top=min_top,
#             width=max_right - min_left,
#             height=max_bottom - min_top,
#         )
#
#     def find_column_boundaries(
#         self, x_positions: List[int], gap_threshold: int = 50
#     ) -> List[Tuple[int, int]]:
#         """
#         Find column boundaries from x-positions
#         """
#         boundaries = []
#
#         if not x_positions:
#             return boundaries
#
#         start = x_positions[0]
#         for i in range(1, len(x_positions)):
#             if x_positions[i] - x_positions[i - 1] > gap_threshold:
#                 boundaries.append((start, x_positions[i - 1] + 20))
#                 start = x_positions[i]
#
#         boundaries.append((start, x_positions[-1] + 20))
#
#         return boundaries
#
#     def extract_table_region(
#         self, keyword_row: pd.Series, ocr_data: pd.DataFrame, context_lines: int = 10
#     ) -> pd.DataFrame:
#         """
#         Extract potential table region around a keyword
#         """
#         top_bound = keyword_row["top"] - 100
#         bottom_bound = keyword_row["top"] + 500
#
#         region = ocr_data[
#             (ocr_data["top"] >= top_bound) & (ocr_data["top"] <= bottom_bound)
#         ]
#
#         return region if len(region) > 5 else None
#
#     def build_table_from_region(self, region: pd.DataFrame) -> Optional[Table]:
#         """
#         Build a table from a detected region
#         """
#         # Similar to build_table_structure but works with a region
#         row_clusters = self.cluster_elements(region, axis="y", threshold=15)
#
#         rows_data = []
#         for row_id in sorted(row_clusters.keys()):
#             row_indices = row_clusters[row_id]
#             row_data = region.iloc[row_indices]
#
#             if len(row_data) >= 2:
#                 col_clusters = self.cluster_elements(row_data, axis="x", threshold=50)
#                 if len(col_clusters) >= 2:
#                     rows_data.append(
#                         {
#                             "row_id": row_id,
#                             "data": row_data,
#                             "col_count": len(col_clusters),
#                             "col_clusters": col_clusters,
#                         }
#                     )
#
#         return self.build_table_structure(rows_data, region) if rows_data else None
#
#     def to_html_enhanced(self, image_path: str) -> str:
#         """
#         Enhanced HTML conversion with better table detection and structure
#         """
#         ocr_data = self.ocr_to_data(image_path)
#
#         # Load image for line detection
#         img = cv2.imread(image_path)
#
#         # Detect tables
#         tables = self.detect_tables_advanced(ocr_data, img)
#
#         # Mark table regions in OCR data
#         table_indices = set()
#         for table in tables:
#             for row in table.cells:
#                 for cell in row:
#                     # Find OCR entries that belong to this table
#                     mask = (
#                         (ocr_data["left"] >= cell.left)
#                         & (ocr_data["left"] <= cell.left + cell.width)
#                         & (ocr_data["top"] >= cell.top)
#                         & (ocr_data["top"] <= cell.top + cell.height)
#                     )
#                     table_indices.update(ocr_data[mask].index.tolist())
#
#         # Build HTML with semantic structure
#         html = [
#             "<!DOCTYPE html>",
#             '<html lang="en">',
#             "<head>",
#             '    <meta charset="UTF-8">',
#             '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
#             "    <title>Invoice OCR Result</title>",
#             "    <style>",
#             "        body { font-family: Arial, sans-serif; margin: 20px; }",
#             "        .invoice-header { font-size: 1.2em; font-weight: bold; margin-bottom: 20px; }",
#             "        .invoice-section { margin-bottom: 15px; }",
#             "        .invoice-table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
#             "        .invoice-table th, .invoice-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
#             "        .invoice-table th { background-color: #f2f2f2; font-weight: bold; }",
#             "        .invoice-table tr:nth-child(even) { background-color: #f9f9f9; }",
#             "        .field-label { font-weight: bold; }",
#             "        .field-value { margin-left: 10px; }",
#             "        .line-item { margin-left: 20px; }",
#             "    </style>",
#             "</head>",
#             "<body>",
#             '    <div class="invoice-container">',
#         ]
#
#         # Process non-table content first
#         non_table_data = ocr_data[~ocr_data.index.isin(table_indices)]
#
#         # Group by blocks and lines for better structure
#         current_section = []
#         last_block = -1
#
#         for block_num in non_table_data["block_num"].unique():
#             block_data = non_table_data[non_table_data["block_num"] == block_num]
#
#             # Check if this is a new section
#             if last_block != -1 and block_num - last_block > 1:
#                 if current_section:
#                     html.append(self.format_section(current_section))
#                     current_section = []
#
#             last_block = block_num
#
#             for line_num in block_data["line_num"].unique():
#                 line_data = block_data[block_data["line_num"] == line_num]
#                 line_text = " ".join([str(text) for text in line_data["text"].values])
#
#                 # Classify line content
#                 line_class = self.classify_line(line_text)
#
#                 current_section.append(
#                     {
#                         "text": line_text,
#                         "class": line_class,
#                         "height": line_data["height"].mean(),
#                     }
#                 )
#
#         # Add remaining section
#         if current_section:
#             html.append(self.format_section(current_section))
#
#         # Add tables with proper structure
#         for i, table in enumerate(tables):
#             html.append(f'        <div class="invoice-section">')
#             html.append(f'            <table class="invoice-table" id="table-{i}">')
#
#             # Determine if first row is header
#             is_header = True
#             for row_idx, row in enumerate(table.cells):
#                 if row_idx == 0 and is_header:
#                     html.append("                <thead>")
#                     html.append("                    <tr>")
#                     for cell in row:
#                         html.append(f"                        <th>{cell.text}</th>")
#                     html.append("                    </tr>")
#                     html.append("                </thead>")
#                     html.append("                <tbody>")
#                 else:
#                     html.append("                    <tr>")
#                     for cell in row:
#                         html.append(f"                        <td>{cell.text}</td>")
#                     html.append("                    </tr>")
#
#             html.append("                </tbody>")
#             html.append("            </table>")
#             html.append("        </div>")
#
#         html.extend(["    </div>", "</body>", "</html>"])
#
#         return "\n".join(html)
#
#     def classify_line(self, text: str) -> str:
#         """
#         Classify line content for better HTML structure
#         """
#         text_lower = text.lower()
#
#         # Headers
#         if any(
#             keyword in text_lower
#             for keyword in ["invoice", "bill", "receipt", "statement"]
#         ):
#             return "header"
#
#         # Field labels
#         if any(
#             keyword in text_lower
#             for keyword in [
#                 "date:",
#                 "invoice #:",
#                 "bill to:",
#                 "ship to:",
#                 "customer:",
#                 "vendor:",
#             ]
#         ):
#             return "field-label"
#
#         # Amounts
#         if re.search(r"[\$€£¥₹]\s*[\d,]+\.?\d*", text) or re.search(
#             r"\d+\.?\d*\s*[\$€£¥₹]", text
#         ):
#             return "amount"
#
#         # Dates
#         if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text) or re.search(
#             r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", text
#         ):
#             return "date"
#
#         return "text"
#
#     def format_section(self, lines: List[Dict]) -> str:
#         """
#         Format a section of non-table content
#         """
#         html = ['        <div class="invoice-section">']
#
#         for line in lines:
#             if line["class"] == "header":
#                 html.append(
#                     f'            <h2 class="invoice-header">{line["text"]}</h2>'
#                 )
#             elif line["class"] == "field-label":
#                 # Try to extract label and value
#                 if ":" in line["text"]:
#                     parts = line["text"].split(":", 1)
#                     label = parts[0].strip()
#                     value = parts[1].strip() if len(parts) > 1 else ""
#                     html.append(
#                         f'            <div><span class="field-label">{label}:</span><span class="field-value">{value}</span></div>'
#                     )
#                 else:
#                     html.append(
#                         f'            <div class="field-label">{line["text"]}</div>'
#                     )
#             elif line["class"] == "amount":
#                 html.append(f'            <div class="amount">{line["text"]}</div>')
#             else:
#                 html.append(f'            <div>{line["text"]}</div>')
#
#         html.append("        </div>")
#
#         return "\n".join(html)
#
#     def create_llm_prompt_enhanced(self, html_content: str) -> str:
#         """
#         Create an enhanced prompt for LLM with better instructions
#         """
#         prompt = """You are an expert invoice data extractor. Analyze the following HTML representation of an invoice and extract ALL information into a structured JSON format.
#
# IMPORTANT INSTRUCTIONS:
# 1. Tables in the HTML likely contain line items with products/services
# 2. Look for patterns in table headers to identify columns (e.g., Description, Quantity, Price, Amount)
# 3. Extract ALL line items found in tables
# 4. Preserve original values and formats (don't convert currencies or dates)
# 5. If information is unclear or missing, use null instead of guessing
#
# REQUIRED JSON STRUCTURE:
# {
#     "invoice_metadata": {
#         "invoice_number": "string or null",
#         "invoice_date": "string (original format) or null",
#         "due_date": "string (original format) or null",
#         "currency": "string (USD, EUR, etc.) or null",
#         "payment_terms": "string or null"
#     },
#     "vendor_info": {
#         "name": "string or null",
#         "address": "string or null",
#         "tax_id": "string or null",
#         "phone": "string or null",
#         "email": "string or null"
#     },
#     "customer_info": {
#         "name": "string or null",
#         "address": "string or null",
#         "tax_id": "string or null",
#         "phone": "string or null",
#         "email": "string or null"
#     },
#     "line_items": [
#         {
#             "item_number": "string or null",
#             "description": "string",
#             "quantity": "number or null",
#             "unit_price": "number or null",
#             "discount": "number or null",
#             "tax_rate": "number or null",
#             "amount": "number"
#         }
#     ],
#     "summary": {
#         "subtotal": "number or null",
#         "discount_total": "number or null",
#         "tax_total": "number or null",
#         "shipping": "number or null",
#         "total_amount": "number",
#         "amount_paid": "number or null",
#         "balance_due": "number or null"
#     },
#     "additional_info": {
#         "notes": "string or null",
#         "terms_and_conditions": "string or null",
#         "bank_details": "string or null"
#     }
# }
#
# HTML CONTENT:
# {}
#
# RESPONSE:
# Return ONLY the JSON object, no explanations or markdown formatting.""".format(
#             html_content
#         )
#
#         return prompt
#
#     def detect_language(self, image_path: str) -> str:
#         """
#         Detect the primary language in the image
#         """
#         try:
#             img = Image.open(image_path)
#             osd = pytesseract.image_to_osd(img)
#             script = re.search(r"Script: (\w+)", osd)
#             if script:
#                 script_name = script.group(1)
#                 script_to_lang = {
#                     "Latin": "eng",
#                     "Cyrillic": "rus",
#                     "Arabic": "ara",
#                     "Han": "chi_sim",
#                     "Hangul": "kor",
#                     "Japanese": "jpn",
#                     "Devanagari": "hin",
#                     "Hebrew": "heb",
#                 }
#                 return script_to_lang.get(script_name, "eng+rus+chi_sim")
#         except:
#             pass
#         return self.languages
#
#
# def process_invoice_enhanced(
#     image_path: str,
#     languages: str = "eng+rus+chi_sim",
#     output_html_path: str = "invoice_output.html",
# ) -> Tuple[str, str]:
#     """
#     Process an invoice with enhanced table detection and HTML conversion
#
#     Args:
#         image_path: Path to invoice image
#         languages: Tesseract language codes
#         output_html_path: Path to save HTML output
#
#     Returns:
#         Tuple of (html_content, llm_prompt)
#     """
#     processor = InvoiceOCRProcessor(languages=languages)
#
#     # Generate enhanced HTML with better table detection
#     html_content = processor.to_html_enhanced(image_path)
#
#     # Create enhanced LLM prompt
#     llm_prompt = processor.create_llm_prompt_enhanced(html_content)
#
#     # Save HTML with proper encoding
#     with open(output_html_path, "w", encoding="utf-8") as f:
#         f.write(html_content)
#
#     print(f"✅ HTML output saved to: {output_html_path}")
#     print(f"📊 Detected tables in the invoice")
#     print(f"📝 LLM prompt generated with enhanced instructions")
#
#     return html_content, llm_prompt
#
#
# def check_tesseract_languages():
#     """
#     Check which language packs are installed for Tesseract
#     """
#     try:
#         langs = pytesseract.get_languages(config="")
#         print("Installed Tesseract languages:")
#         for lang in langs:
#             print(f"  - {lang}")
#         return langs
#     except Exception as e:
#         print(f"Error checking languages: {e}")
#         return []
#
#
# if __name__ == "__main__":
#     # Check installed languages
#     print("Checking Tesseract configuration...")
#     installed_langs = check_tesseract_languages()
#     print("\n" + "=" * 50 + "\n")
#
#     # Process invoice
#     invoice_path = "invoice_image.png"  # Replace with your invoice path
#
#     print(f"Processing invoice: {invoice_path}")
#     html_content, llm_prompt = process_invoice_enhanced(
#         image_path=invoice_path,
#         languages="eng+rus+chi_sim",  # Adjust based on your needs
#         output_html_path="enhanced_invoice.html",
#     )
#
#     print("\n" + "=" * 50)
#     print(llm_prompt)
#
#     # Optional: Send to LLM for extraction
#     # extracted_data = send_to_llm(llm_prompt)
#     # print(json.dumps(extracted_data, indent=2))
