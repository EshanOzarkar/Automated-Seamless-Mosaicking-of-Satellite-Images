import cv2
import os
from skimage import exposure
from skimage.transform import resize

# Function for radiometric normalization
def radiometric_normalization(image):
    normalized_image = exposure.equalize_adapthist(image)
    return normalized_image

# Function to resize an image
def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# Function to process a single image
def process_image(image_file, input_folder, output_folder):
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image (optional, adjust scale_percent as needed)
    image = resize_image(image, scale_percent=50)
    
    # Perform radiometric normalization
    normalized_image = radiometric_normalization(image)
    
    # Convert normalized_image to the expected range and data type
    normalized_image_255 = (normalized_image * 255).astype(image.dtype)
    
    # Save the normalized image to the output folder
    cv2.imwrite(output_path, normalized_image_255)
    print(f"Processed {image_file}")

# Main function to orchestrate the parallel processing
def main():
    input_folder = "SS607_Fully_Automated"
    output_folder = "r_output"
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.png', '.jpeg', '.tif'))]
    
    for image_file in image_files:
        process_image(image_file, input_folder, output_folder)

    print("Radiometric normalization completed. Normalized images are stored in the 'r_output' folder.")

if __name__ == "__main__":
    main()
