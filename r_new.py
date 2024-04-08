import cv2
import numpy as np
from skimage import exposure
import psutil
import os

# Function for radiometric normalization
def radiometric_normalization(image):
    # Separate color channels
    r, g, b = cv2.split(image)

    # Apply normalization to each channel separately
    r_normalized = exposure.equalize_adapthist(r)
    g_normalized = exposure.equalize_adapthist(g)
    b_normalized = exposure.equalize_adapthist(b)

    # Merge normalized channels
    normalized_image = cv2.merge((r_normalized, g_normalized, b_normalized))

    return normalized_image

# Path to the input image folder and output folder
input_folder_path = "SS607_Fully_Automated"
output_folder_path = "r_out_new_color"  # Output folder for colored images

# Get available RAM in GB
available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

# Function to process a single image
def process_single_image(input_image_path, output_folder_path, available_ram_gb):
    # Read the input image in color mode
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    # Check if the image is successfully loaded
    if input_image is not None:
        # Perform radiometric normalization
        normalized_image = radiometric_normalization(input_image)

        # Ensure the normalized image is in the appropriate data range
        normalized_image = (normalized_image * 255).astype(np.uint8)

        # Create the output path for the normalized image
        output_image_path = os.path.join(output_folder_path, os.path.basename(input_image_path)[:-4] + "_normalized.tif")

        # Save the normalized image
        cv2.imwrite(output_image_path, normalized_image)

        print("Radiometric normalization completed. Normalized image is saved at:", output_image_path)
    else:
        print(f"Error: Could not read the image file at {input_image_path}")

# Process each image in the input folder
input_files = os.listdir(input_folder_path)
for input_file in input_files:
    # Check if the file is an image
    if input_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        # Path to the input image
        input_image_path = os.path.join(input_folder_path, input_file)

        # Process the image
        process_single_image(input_image_path, output_folder_path, available_ram_gb)
