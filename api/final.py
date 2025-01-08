import cv2
import numpy as np
from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode
from PIL import Image
import io
import pytesseract
from flask_cors import CORS
import easyocr  # For handwritten OCR

import torch

app = Flask(__name__)
CORS(app)
# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # English language support


def detect_and_crop_barcode(image, margin_mm=2, dpi=300, min_width=100, min_height=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barcode_contour = None

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            barcode_contour = approx
            break

    if barcode_contour is not None:
        x, y, w, h = cv2.boundingRect(barcode_contour)
        
        # Check if the bounding box dimensions are large enough
        if w < min_width or h < min_height:
            return None  # Ignore small contours

        margin_pixels = int((margin_mm / 25.4) * dpi)
        x_new = max(0, x + margin_pixels)
        y_new = max(0, y + margin_pixels)
        w_new = max(1, w - 2 * margin_pixels)
        h_new = max(1, h - 2 * margin_pixels)
        cropped_barcode = image[y_new:y_new + h_new, x_new:x_new + w_new]

        return cropped_barcode

    return None


# Extract handwritten digits using EasyOCR (New Fallback)
def extract_handwritten_digits(image):
    # Convert image to grayscale for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to extract text
    results = reader.readtext(gray, detail=0)  # Only fetch text, ignore box details

    # Filter digits from the detected text
    digits = ''.join([res for res in results if res.isdigit()])
    return digits



@app.route('/scan-barcode', methods=['POST'])
def scan_barcode():
    result_array = []
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    print(f"File received: {file.filename}, Content-Type: {file.content_type}")

    # Read the image from the uploaded file
    image = Image.open(io.BytesIO(file.read()))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Try detecting and decoding barcode
    cropped_barcode = detect_and_crop_barcode(image_cv)
    if cropped_barcode is not None:
        print("dealing with cropped image")
        barcodes = decode(Image.fromarray(cropped_barcode))
        if barcodes:
            result_array.extend([{"data": barcode.data.decode("utf-8"), "type": barcode.type} for barcode in barcodes])

    # Fallback to decoding without cropping
    barcodes = decode(image)
    if barcodes:
        result_array.extend([{"data": barcode.data.decode("utf-8"), "type": barcode.type} for barcode in barcodes])

    # Try Handwritten Digit Extraction
    print("Trying handwritten digit extraction...")
    digits = extract_handwritten_digits(image_cv)
    if digits:
        print(f"Handwritten digits detected: {digits}")
        result_array.append({"barcode": digits})

    # Return results if any were found
    if result_array:
        return jsonify(result_array)

    return jsonify({"message": "No barcode or digits detected"}), 404


if __name__ == '__main__':
    app.run(debug=True)
