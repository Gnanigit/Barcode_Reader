from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import io
import pytesseract
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    """Preprocess the image to prepare for barcode region detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to handle lighting variations
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to close gaps and enhance structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

def find_barcode_region(image):
    """Locate the barcode region in the image."""
    processed_image = preprocess_image(image)

    # Find contours in the processed image
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barcode_region = None

    if contours:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(c)

            # Aspect ratio and size filtering
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 8.0 and w * h > 5000:  # Refine thresholds as needed
                barcode_region = image[y:y+h, x:x+w]
                return barcode_region

    return None

def refine_barcode_region(barcode_region):
    """Remove external boundaries and noise from the barcode region."""
    gray = cv2.cvtColor(barcode_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assumed to be the barcode itself
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the barcode region to the largest contour
        barcode_region_refined = barcode_region[y:y+h, x:x+w]
        return barcode_region_refined

    return barcode_region

def extract_text_below_barcode(barcode_region):
    """Use OCR to extract the text (numeric value) below the barcode."""

    gray = cv2.cvtColor(barcode_region, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    cropped = gray[int(height * 0.8):, :] 
    _, binary = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)

    config = '--psm 6 -c tessedit_char_whitelist=0123456789'  
    text = pytesseract.image_to_string(binary, config=config).strip()

    return text

@app.route('/scan-barcode', methods=['POST'])
def scan_barcode():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    barcodes = decode(image)
    if barcodes:
        result = [{"data": barcode.data.decode("utf-8"), "type": barcode.type} for barcode in barcodes]
        return jsonify({"barcodes": result})

    barcode_region = find_barcode_region(image_cv)
    if barcode_region is not None:
        barcode_region_refined = refine_barcode_region(barcode_region)
        barcodes = decode(Image.fromarray(barcode_region_refined))

        if barcodes:
            result = [{"data": barcode.data.decode("utf-8"), "type": barcode.type} for barcode in barcodes]

            ocr_text = extract_text_below_barcode(barcode_region_refined)
            if ocr_text:
                result.append({"ocr_text": ocr_text})

            return jsonify({"barcodes": result})

        ocr_text = extract_text_below_barcode(barcode_region_refined)
        if ocr_text:
            return jsonify({"message": "Barcode not decoded, but OCR text extracted", "ocr_text": ocr_text})

    return jsonify({"message": "No barcode detected"}), 404

if __name__ == '__main__':
    app.run(debug=True)
