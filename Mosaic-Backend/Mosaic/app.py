from flask import Flask, request, send_file, render_template
import os
import cv2
import tempfile
import shutil

from radiometric_normalization import radiometric_normalization
from orthorectify import orthorectify_images
from match_images import determine_order
from stitch_images import stitch_images

def load_images(folder_path):
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    images = [cv2.imread(file) for file in image_files if cv2.imread(file) is not None]
    return images, image_files

def pipeline(input_folder_path, output_folder_path = "output"):
    # Load images
    images, image_files = load_images(input_folder_path)

    # Step 1: Radiometric normalization
    normalized_images = radiometric_normalization(images)

    # Step 2: Orthorectification
    orthorectified_images = orthorectify_images(normalized_images)

    # Step 3: Determine image order based on matching
    order = determine_order(orthorectified_images)
    
    # Correcting the usage of order for ordering the images
    ordered_images = [orthorectified_images[i] for i in order]

    # Step 4: Stitch images
    stitched_image = stitch_images(ordered_images)
    stitched_image_path = os.path.join(output_folder_path, 'final_stitched_image.jpg')
    if stitched_image is not None:
        cv2.imwrite(stitched_image_path, stitched_image)
        print(f"Final stitched image saved to {stitched_image_path}")
    else:
        print("Stitching failed.")

if __name__ == "__main__":
    input_folder_path = input("Enter the path to the folder containing images: ")
    pipeline(input_folder_path)
