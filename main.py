# This is my Python OCR project to convert my D&D books and magazines from images to text, so I can both save space
# and parse the data within more easily in preparation for database and ML.

import cv2
import pytesseract
from PIL import Image
import os
import numpy as np
from spellchecker import SpellChecker
import language_tool_python

# ====== CONFIGURATION ======

# Path to Tesseract executable (update if yours is different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Image file path
image_path = "data/PHB_Pg4.jpg"

# Output paths
processed_image_path = "data/processed_PHB.jpg"
ocr_output_path = "data/ocr_output.txt"

# Tesseract config
custom_config = r'--oem 3 --psm 3'

# ====== FILE CHECK ======

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image file not found: {image_path}")

# ====== IMAGE PREPROCESSING ======

# Step 1: Load image
cv_image = cv2.imread(image_path)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply thresholding
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Step 4: Resize to improve accuracy
resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Step 5: Denoise the image
denoised = cv2.fastNlMeansDenoising(resized, h=30)

# Step 6: Sharpen the image
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(denoised, -1, kernel)

# Step 7: Save processed image for inspection
cv2.imwrite(processed_image_path, sharpened)

# Optional: Preview the image
# cv2.imshow("Processed", sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====== OCR PROCESS ======

ocr_result = pytesseract.image_to_string(sharpened, config=custom_config)

# ====== SPELLCHECK + POSTPROCESSING ======

spell = SpellChecker()
words = ocr_result.split()
corrected = [
    spell.correction(word) if spell.correction(word) is not None else word
    for word in words
]
corrected_text = " ".join(corrected)

# Initialize the language tool (English)
tool = language_tool_python.LanguageTool('en-US')

# Run grammar and context-aware corrections
matches = tool.check(corrected_text)
final_text = language_tool_python.utils.correct(corrected_text, matches)

# Print final output
print("\n📜 Final OCR Output (Corrected with Grammar Tool):\n")
print(final_text)

# Save to file
with open(ocr_output_path, "w", encoding="utf-8") as f:
    f.write(final_text)

