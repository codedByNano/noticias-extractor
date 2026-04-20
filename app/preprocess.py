from pdf2image import convert_from_path
import os
import cv2
import numpy as np

OUTPUT_DIR = "data/pages"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def pdf_to_images(pdf_path: str):
    ensure_dirs()

    pages = convert_from_path(pdf_path, dpi=200)

    image_paths = []

    for i, page in enumerate(pages):
        img = normalize_image(page)
        path = os.path.join(OUTPUT_DIR, f"page_{i}.jpg")
        cv2.imwrite(path, img)
        image_paths.append(path)

    return image_paths
