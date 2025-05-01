# This is my Python OCR project to convert my D&D books and magazines from images to text, so I can both save space
# and parse the data within more easily in preparation for database and ML.

# import cv2
from PIL import Image
# import pytesseract

im_file = "venv/data/PHB_Test_Pg4.jpg"

im = Image.open(im_file)
# im.show()  to show in Windows Viewer





