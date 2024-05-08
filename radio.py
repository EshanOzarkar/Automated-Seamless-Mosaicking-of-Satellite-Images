import cv2
import numpy as np
from skimage import exposure
import os

def radiometric_normalization(image):
    r, g, b = cv2.split(image)
    r_normalized = exposure.equalize_adapthist(r)
    g_normalized = exposure.equalize_adapthist(g)
    b_normalized = exposure.equalize_adapthist(b)
    normalized_image = cv2.merge((r_normalized, g_normalized, b_normalized))
    return normalized_image

def process_single_image(input_image_path, output_folder_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if input_image is not None:
        normalized_image = radiometric_normalization(input_image)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        output_image_path = os.path.join(output_folder_path, os.path.basename(input_image_path)[:-4] + "_normalized.tif")
        cv2.imwrite(output_image_path, normalized_image)
        print("Radiometric normalization completed. Normalized image is saved at:", output_image_path)
    else:
        print(f"Error: Could not read the image file at {input_image_path}")

def process_images_for_radiometric_normalization(input_folder_path, output_folder_path):
    # Ensure output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    input_files = os.listdir(input_folder_path)
    for input_file in input_files:
        if input_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            input_image_path = os.path.join(input_folder_path, input_file)
            process_single_image(input_image_path, output_folder_path)

    return output_folder_path
