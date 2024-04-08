import os
import cv2
import numpy as np

# Function for orthorectification (using OpenCV for perspective transformation)
def apply_orthorectification_transformation(image, georeference_data):
    # Extract geotransform parameters
    geotransform = georeference_data["geotransform"]
    # Create a 2x3 transformation matrix from the geotransform parameters
    transformation_matrix = np.array([[geotransform[0], geotransform[1], geotransform[2]],
                                      [geotransform[3], geotransform[4], geotransform[5]]], dtype=np.float32)
    # Apply perspective transformation
    orthorectified_image = cv2.warpAffine(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return orthorectified_image

# Function for orthorectification
def orthorectify_image(image, georeference_data):
    # Implement orthorectification using the provided georeference_data
    orthorectified_image = apply_orthorectification_transformation(image, georeference_data)
    return orthorectified_image

# Example georeference_data
georeference_data_example = {"geotransform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}

# Input and output folders
input_folder = "r_out_new_color"
output_folder = "o_out"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".jpg"): # Adjust based on the image format you have
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Perform orthorectification with the example georeference_data
        orthorectified_image = orthorectify_image(image, georeference_data_example)
        
        # Save the orthorectified image
        output_image_path_orthorectified = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path_orthorectified, orthorectified_image)
        print("Orthorectification completed for:", filename)

print("All images orthorectified and saved to:", output_folder)
