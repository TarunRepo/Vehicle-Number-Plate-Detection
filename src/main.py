import cv2
import imutils
import numpy as np
import pytesseract
import argparse
import sys

# Configure Tesseract path if necessary (e.g., for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.ext'

class LicensePlateDetector:
    def __init__(self, min_area=500):
        self.min_area = min_area
        
    def preprocess_image(self, image):
        """Convert to grayscale, blur, and find edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to remove noise while keeping edges sharp
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
        # Canny edge detection
        edged = cv2.Canny(bfilter, 30, 200) 
        return gray, edged

    def find_plate_contour(self, edged):
        """Find the contour that most likely represents a license plate."""
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        # Sort contours by area in descending order and keep top 10
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            # A license plate generally has 4 corners
            if len(approx) == 4:
                location = approx
                break
        return location

    def extract_text(self, gray_image, location):
        """Extract the license plate from the image and apply OCR."""
        if location is None:
            return None, None
            
        mask = np.zeros(gray_image.shape, np.uint8)
        # Draw the contour on the mask
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
        # Crop the image to the bounding box of the license plate
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped_plate = gray_image[topx:bottomx+1, topy:bottomy+1]
        
        # Apply adaptive thresholding to improve OCR accuracy
        thresh = cv2.adaptiveThreshold(cropped_plate, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Read text using Tesseract OCR
        # --psm 8: Treat the image as a single word (good for license plates)
        text = pytesseract.image_to_string(thresh, config='--psm 8')
        return text.strip(), cropped_plate

def main():
    parser = argparse.ArgumentParser(description="Vehicle Number Plate Detection System")
    parser.add_argument("-i", "--image", type=str, help="Path to input image")
    args = parser.parse_args()

    # For demonstration, if no image is provided, exit gracefully
    if not args.image:
        print("[INFO] No input image provided. Usage: python main.py --image path/to/car.jpg")
        print("[INFO] Simulating pipeline initialization...")
        detector = LicensePlateDetector()
        print("[SUCCESS] Pipeline initialized. Ready for video stream or image input.")
        sys.exit(0)

    print(f"[INFO] Loading image {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        print("[ERROR] Could not read image.")
        sys.exit(1)

    detector = LicensePlateDetector()
    
    print("[INFO] Preprocessing image (Grayscale + Canny Edge Detection)...")
    gray, edged = detector.preprocess_image(img)
    
    print("[INFO] Detecting contours...")
    location = detector.find_plate_contour(edged)
    
    if location is not None:
        print("[INFO] License plate contour found. Applying OCR...")
        text, plate_img = detector.extract_text(gray, location)
        print(f"\n[RESULT] Detected License Plate: {text}\n")
        
        # Draw bounding box and text on original image
        cv2.drawContours(img, [location], -1, (0, 255, 0), 3)
        # Display the result (requires GUI/Display environment)
        # cv2.imshow("Plate", plate_img)
        # cv2.imshow("Result", img)
        # cv2.waitKey(0)
    else:
        print("[WARN] No license plate contour detected in the image.")

if __name__ == "__main__":
    main()
