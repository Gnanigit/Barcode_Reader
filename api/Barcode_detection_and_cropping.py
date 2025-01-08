import cv2
import numpy as np

def detect_and_crop_barcode(input_image_path, output_image_path="cropped_barcode.jpg", margin_mm=5, dpi=300):
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError("Error: Unable to load the image. Please check the path.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the barcode contour
    barcode_contour = None

    # Loop through the contours to find the barcode
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Look for quadrilateral contours with a significant area
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            barcode_contour = approx
            break

    if barcode_contour is not None:
        # Get bounding box coordinates for the detected barcode
        x, y, w, h = cv2.boundingRect(barcode_contour)

        # Convert margin from mm to pixels
        margin_pixels = int((margin_mm / 25.4) * dpi)

        # Adjust the bounding box to crop further inside
        x_new = x + margin_pixels
        y_new = y + margin_pixels
        w_new = max(0.5, w -  margin_pixels)  # Ensure width is at least 1 pixel
        h_new = max(0.5, h - margin_pixels)  # Ensure height is at least 1 pixel

        # Crop the barcode region from the image with the inner margin
        cropped_barcode = image[y_new:y_new+h_new, x_new:x_new+w_new]

        # Save the cropped barcode image
        cv2.imwrite(output_image_path, cropped_barcode)
        print(f"Cropped barcode saved at: {output_image_path}")
    else:
        print("No barcode detected.")

# Call the function with a margin of 5mm
detect_and_crop_barcode("input.jpg", "cropped_barcode.jpg", margin_mm=1, dpi=300)
