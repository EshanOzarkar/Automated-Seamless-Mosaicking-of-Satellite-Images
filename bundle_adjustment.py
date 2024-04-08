import cv2
import numpy as np
from scipy.optimize import least_squares
from functools import partial
import os

def apply_adjustment(image, optimized_params):
    rotation_vector = optimized_params[:3]
    translation_vector = optimized_params[3:]
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformation_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    adjusted_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return adjusted_image

# Function to perform bundle adjustment for a single image pair
def bundle_adjustment_single(images, params_flat):
    num_images = len(images)
    print(num_images)
    num_params_per_image = 6  # Assuming 6 parameters per image (3 for rotation, 3 for translation)

    # Reshape the flattened parameters for this image pair
    params = params_flat.reshape((num_params_per_image,))

    # Compute the transformation matrix using the parameters
    transformation_matrix = compute_transformation_matrix(params)

    # Apply the transformation to the second image
    warped_image = cv2.warpPerspective(images[1], transformation_matrix, (images[1].shape[1], images[1].shape[0]))

    # Compute the residual between the warped image and the first image
    residual = images[0] - warped_image

    # Flatten the residual for optimization
    return residual.flatten()

# Function to compute transformation matrix from parameters
def compute_transformation_matrix(params):
    # Example: compute transformation matrix from rotation and translation parameters
    rotation_matrix = cv2.Rodrigues(params[:3])[0]
    translation_vector = params[3:]
    transformation_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    return transformation_matrix

# Load feature-based registered images from a folder (replace with your own image loading code)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Example usage
folder_path = "f_out"
output_folder = "b_out"  # Define the output folder

images = load_images_from_folder(folder_path)

# Define initial parameters (e.g., camera parameters, rotation, translation)
initial_params = np.zeros((len(images), 6))  # Assuming 6 parameters per image (3 for rotation, 3 for translation)
initial_params[0] = [0, 0, 0, 0, 0, 0]  # Initialize the first image with identity transformation

# Flatten the initial parameters for optimization
initial_params_flat = initial_params.flatten()

# Define the residual function for bundle adjustment for the first image pair
residual_func_first_pair = partial(bundle_adjustment_single, images[:2])

# Perform bundle adjustment for the first image pair using least squares optimization
result_first_pair = least_squares(residual_func_first_pair, initial_params_flat, method='lm')

# Reshape the optimized parameters for the first image pair
optimized_params_first_pair = result_first_pair.x.reshape((-1, 6))

# Apply the optimized parameters to the second image in the pair
adjusted_image_second = apply_adjustment(images[1], optimized_params_first_pair[1])

# Now you have the adjusted second image ready for further processing

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the adjusted second image to the output folder
output_image_path_second = os.path.join(output_folder, "adjusted_image_1.tif")
cv2.imwrite(output_image_path_second, adjusted_image_second)

# Continue performing bundle adjustment for subsequent image pairs in a loop
for i in range(2, len(images)):
    # Define the residual function for bundle adjustment for the current image pair
    residual_func_current_pair = partial(bundle_adjustment_single, [adjusted_image_second, images[i]])

    # Perform bundle adjustment for the current image pair using least squares optimization
    result_current_pair = least_squares(residual_func_current_pair, optimized_params_first_pair[1], method='lm')

    # Reshape the optimized parameters for the current image pair
    optimized_params_current_pair = result_current_pair.x.reshape((-1, 6))

    # Apply the optimized parameters to the current image in the pair
    adjusted_image_current = apply_adjustment(images[i], optimized_params_current_pair[0])

    # Save the adjusted current image to the output folder
    output_image_path_current = os.path.join(output_folder, f"adjusted_image_{i}.tif")
    cv2.imwrite(output_image_path_current, adjusted_image_current)

    # Update the adjusted image for the next iteration
    adjusted_image_second = adjusted_image_current

print("Bundle Adjustment completed. Adjusted images are saved in:", output_folder)
