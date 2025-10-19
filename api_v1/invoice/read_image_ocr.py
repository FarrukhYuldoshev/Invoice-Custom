import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from typing import Dict, List, Any, Optional
import pandas as pd


class InvoiceOCRProcessor:
    def __init__(
        self, tesseract_cmd: Optional[str] = None, languages: str = "eng+rus+chi_sim"
    ):
        """
        Initialize the processor

        Args:
            tesseract_cmd: Path to tesseract executable if not in PATH
            languages: Tesseract language codes (e.g., 'eng+rus+chi_sim' for multiple languages)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.languages = languages

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get better OCR results
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(thresh, 1)

        # Resize if image is too small
        height, width = denoised.shape
        if width < 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            denoised = cv2.resize(
                denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

        return denoised

    def detect_language(self, image_path: str) -> str:
        """
        Detect the primary language in the image

        Args:
            image_path: Path to the image file

        Returns:
            Detected language code
        """
        try:
            img = Image.open(image_path)
            # Try to detect language
            osd = pytesseract.image_to_osd(img)
            script = re.search(r"Script: (\w+)", osd)
            if script:
                script_name = script.group(1)
                # Map scripts to language codes
                script_to_lang = {
                    "Latin": "eng",
                    "Cyrillic": "rus",
                    "Arabic": "ara",
                    "Han": "chi_sim",
                    "Hangul": "kor",
                    "Japanese": "jpn",
                }
                return script_to_lang.get(script_name, "eng+rus+chi_sim")
        except:
            pass
        return self.languages

    def ocr_to_data(self, image_path: str, preprocess: bool = True) -> pd.DataFrame:
        """
        Extract text with layout information using Tesseract

        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image

        Returns:
            DataFrame with OCR results including position information
        """
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            img = cv2.imread(image_path)

        # Detect or use specified languages
        lang = (
            self.detect_language(image_path)
            if hasattr(self, "auto_detect")
            else self.languages
        )

        # Get detailed OCR data with proper language support
        custom_config = f"-l {lang} --oem 3 --psm 6 -c preserve_interword_spaces=1"

        # Ensure UTF-8 encoding for proper character handling
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DATAFRAME,
            config=custom_config,
            lang=lang,
        )

        # Filter out empty text and decode properly
        data = data[data["text"].notna() & (data["text"] != "")]

        # Ensure text is properly encoded
        data["text"] = data["text"].apply(
            lambda x: str(x).encode("utf-8", errors="ignore").decode("utf-8")
        )

        return data

    def detect_tables(self, ocr_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Detect and extract potential tables from OCR data

        Args:
            ocr_data: DataFrame with OCR results

        Returns:
            List of DataFrames representing detected tables
        """
        tables = []

        # Group text by lines based on y-coordinates
        if len(ocr_data) == 0:
            return tables

        ocr_data["line_group"] = pd.cut(ocr_data["top"], bins=20, labels=False)

        # Identify potential table regions (lines with multiple aligned elements)
        for line_num in ocr_data["line_group"].unique():
            line_data = ocr_data[ocr_data["line_group"] == line_num]
            if len(line_data) >= 3:  # Potential table row
                # Check for column alignment
                x_positions = sorted(line_data["left"].values)
                if len(x_positions) >= 3:
                    # Simple heuristic: consistent spacing suggests table
                    spacing = [
                        x_positions[i + 1] - x_positions[i]
                        for i in range(len(x_positions) - 1)
                    ]
                    if max(spacing) < 500:  # Reasonable column spacing
                        tables.append(line_data)

        return tables

    def to_html(self, image_path: str) -> str:
        """
        Convert invoice to HTML format preserving layout and encoding

        Args:
            image_path: Path to the image file

        Returns:
            HTML string representation with UTF-8 encoding
        """
        ocr_data = self.ocr_to_data(image_path)

        html = ["<!DOCTYPE html>"]
        html.append("<html>")
        html.append("<head>")
        html.append('<meta charset="UTF-8">')
        html.append(
            '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
        )
        html.append("</head>")
        html.append("<body>")
        html.append('<div class="invoice">')

        # Group by block for better structure
        for block_num in ocr_data["block_num"].unique():
            block_data = ocr_data[ocr_data["block_num"] == block_num]

            html.append('<div class="block">')

            # Group by line
            for line_num in block_data["line_num"].unique():
                line_data = block_data[block_data["line_num"] == line_num]
                # Ensure proper encoding for each text element
                line_text = " ".join([str(text) for text in line_data["text"].values])

                # Check if this might be a header (usually larger font or bold)
                avg_height = line_data["height"].mean()
                if avg_height > ocr_data["height"].mean() * 1.2:
                    html.append(f"<h2>{line_text}</h2>")
                else:
                    html.append(f"<p>{line_text}</p>")

            html.append("</div>")

        # Detect and add tables
        tables = self.detect_tables(ocr_data)
        if tables:
            html.append('<div class="tables">')
            for table in tables:
                html.append("<table>")
                html.append("<tr>")
                for _, row in table.iterrows():
                    html.append(f'<td>{str(row["text"])}</td>')
                html.append("</tr>")
                html.append("</table>")
            html.append("</div>")

        html.append("</div>")
        html.append("</body></html>")

        return "\n".join(html)

    def create_llm_prompt(
        self, structured_content: str, output_format: str = "json"
    ) -> str:
        """
        Create a prompt for the LLM to extract structured data
        Args:
            structured_content: The structured text/html/xml content
            output_format: Desired output format
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""Extract all information from the following invoice and convert it to {output_format} format.

Include the following fields if present:
- invoice_number
- invoice_date
- due_date
- vendor_name
- vendor_address
- vendor_tax_id
- customer_name
- customer_address
- customer_tax_id
- line_items (array with: description, quantity, unit_price, amount)
- subtotal
- tax_rate
- tax_amount
- total_amount
- currency
- payment_terms
- notes

Invoice Content:
{structured_content}

Output only valid {output_format} without any explanation."""

        return prompt


def process_invoice_multilang(image_path: str, languages: str = "eng+rus+chi_sim"):
    """
    Process an invoice with multi-language support
    Args:
        image_path: Path to invoice image
        languages: Tesseract language codes ('eng+rus' for English and Russian)

    Returns:
        Tuple of (structured_content, llm_prompt)
    """
    # Initialize with proper language support
    processor = InvoiceOCRProcessor(languages=languages)

    content = processor.to_html(image_path)
    prompt: str = processor.create_llm_prompt(content)

    # Save with proper encoding if needed
    with open("output.html", "w", encoding="utf-8") as f:
        f.write(content)

    return content, prompt


# Installation helper for language packs
def check_tesseract_languages():
    """
    Check which language packs are installed for Tesseract
    """
    try:
        langs = pytesseract.get_languages(config="")
        print("Installed Tesseract languages:")
        for lang in langs:
            print(f"  - {lang}")
        return langs
    except Exception as e:
        print(f"Error checking languages: {e}")
        return []


if __name__ == "__main__":
    installed_langs = check_tesseract_languages()
    invoice_path = "img_1.png"  # Replace with your invoice path
    content, prompt = process_invoice_multilang(image_path=invoice_path)
    print(prompt)
    # print("\nOutput saved with proper UTF-8 encoding!")
