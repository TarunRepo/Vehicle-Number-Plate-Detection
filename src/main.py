import cv2
import pytesseract
import os

# ==============================
# TESSERACT PATH
# ==============================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
    print("Tesseract not found!")
    exit()

# ==============================
# LOAD IMAGE
# ==============================

image_path = "assets/car_plate.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

# Create copy
output = img.copy()

# ==============================
# CONVERT TO GRAYSCALE
# ==============================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==============================
# LOAD HAAR CASCADE
# ==============================

cascade_path = "assets/haarcascade_russian_plate_number.xml"

plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("Could not load cascade file!")
    exit()

# ==============================
# DETECT NUMBER PLATES
# ==============================

plates = plate_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(80, 20)
)

print(f"Number Plates Found: {len(plates)}")

# ==============================
# PROCESS EACH PLATE
# ==============================

for i, (x, y, w, h) in enumerate(plates):

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    # Extract ROI
    plate_roi = gray[y:y+h, x:x+w]

    # Resize ROI for better OCR
    plate_roi = cv2.resize(
        plate_roi,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Blur
    plate_roi = cv2.GaussianBlur(
        plate_roi,
        (5, 5),
        0
    )

    # Threshold
    _, plate_roi = cv2.threshold(
        plate_roi,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR
    text = pytesseract.image_to_string(
        plate_roi,
        config='--psm 8'
    )

    text = text.strip()

    print(f"\nPlate {i+1}:")
    print(text)

    # Display text on image
    cv2.putText(
        output,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(f"Plate ROI {i+1}", plate_roi)

# ==============================
# SHOW RESULTS
# ==============================

cv2.imshow("Detected Number Plate", output)

cv2.waitKey(0)
cv2.destroyAllWindows()