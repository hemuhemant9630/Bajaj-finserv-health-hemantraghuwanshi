import pytesseract
import cv2
import numpy as np
import tempfile
import re
from PIL import Image
import io
import os  # Add missing import

def process_lab_report(image_bytes):
    temp_path = None  # Ensure temp_path is defined
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        if not os.path.exists(temp_path):
            raise Exception("Failed to save temporary image file")

        # Improve image preprocessing
        img = cv2.imread(temp_path)
        # Resize image to improve OCR
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Denoise image
        denoised = cv2.fastNlMeansDenoising(thresh)
    
        # Configure tesseract parameters
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(denoised, config=custom_config)
        
        # Split into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
    
        results = []
        for line in lines:
            # Try different patterns
            patterns = [
                # Pattern for test with value and range
                r"([A-Za-z\s\(\).-]+?)\s*([-]?\d+\.?\d*)\s*([A-Za-z/%]+)?\s*([-]?\d+\.?\d*\s*-\s*[-]?\d+\.?\d*)?",
                # Pattern for positive/negative results
                r"([A-Za-z\s\(\).-]+?)\s*(POSITIVE|NEGATIVE)\s*([A-Za-z/%]+)?",
                # Pattern for test name followed by value on next line
                r"([A-Za-z\s\(\).-]+?)\s*:\s*([-]?\d+\.?\d*)",
            ]
    
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    test_name = match.group(1).strip()
                    value = match.group(2)
                    
                    # Skip if test name contains unwanted words
                    if any(word in test_name.lower() for word in ['date', 'no.', 'name', 'doctor', 'mobile']):
                        continue
                    
                    # Handle units and reference range
                    unit = match.group(3) if len(match.groups()) > 2 and match.group(3) else "-"
                    ref_range = match.group(4) if len(match.groups()) > 3 and match.group(4) else "-"
                    
                    # Determine if out of range
                    out_of_range = False
                    if ref_range != "-" and ref_range:
                        try:
                            ref_low, ref_high = map(float, ref_range.replace(" ", "").split("-"))
                            value_float = float(value)
                            out_of_range = not (ref_low <= value_float <= ref_high)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(value, str) and value.upper() == "POSITIVE":
                        out_of_range = True
                    
                    result = {
                        "test_name": test_name.upper(),
                        "test_value": value,
                        "bio_reference_range": ref_range if ref_range != "-" else "-",
                        "test_unit": unit if unit != "-" else "-",
                        "lab_test_out_of_range": out_of_range
                    }
                    
                    if result not in results:
                        results.append(result)
                    break
    
        # Debug output
        print("Extracted text:", text)
        print("Processed results:", results)
        
        return {
            "is_success": True,
            "data": results
        }
    except Exception as e:
        print(f"Error in process_lab_report: {str(e)}")
        raise Exception(f"Failed to process lab report: {str(e)}")
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"Error cleaning up temp file: {cleanup_err}")
