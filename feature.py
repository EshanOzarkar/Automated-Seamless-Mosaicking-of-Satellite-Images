import cv2
import numpy as np
import os

# Function for feature-based registration with incremental approach
def incremental_registration(images, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    reference_index = find_reference_image(images)
    reference_image = cv2.imread(images[reference_index])
    reference_keypoints, reference_descriptors = detect_features(reference_image)
    registered_images = [reference_image]

    # Loop through the images starting from the second one
    for i in range(1, len(images)):
        # Load the current image
        current_image = cv2.imread(images[i])
        
        # Detect keypoints and descriptors
        keypoints_curr, descriptors_curr = detect_features(current_image)

        # Match keypoints between current image and reference image
        matches = match_features(reference_keypoints, reference_descriptors, keypoints_curr, descriptors_curr)

        # Compute registration transformation
        transformation_matrix = compute_registration_transform(matches, reference_keypoints, keypoints_curr)

        # Apply transformation to the current image
        registered_image = cv2.warpPerspective(current_image, transformation_matrix, (reference_image.shape[1], reference_image.shape[0]))

        # Save the registered image to the output folder
        output_path = os.path.join(output_folder, f"registered_image_{i}.png")
        cv2.imwrite(output_path, registered_image)
        print(f"Registered image saved at: {output_path}")

        # Update the reference keypoints and descriptors
        reference_keypoints, reference_descriptors = keypoints_curr, descriptors_curr

        # Append the registered image to the list
        registered_images.append(registered_image)

        # Release memory for the current image
        del current_image

    return registered_images

# Function to find the image with the highest quality keypoints and descriptors
def find_reference_image(images):
    best_score = -1
    best_index = 0
    for i, image_path in enumerate(images):
        # Load the image and detect keypoints
        image = cv2.imread(image_path)
        keypoints, descriptors = detect_features(image)
        
        # Calculate score based on the quality of keypoints and descriptors
        score = calculate_score(keypoints, descriptors)
        if score > best_score:
            best_score = score
            best_index = i

        # Release memory for the image
        del image

    return best_index

# Example functions (replace with actual feature detection and matching algorithms)
def detect_features(image):
    # Example feature detection using SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(keypoints1, descriptors1, keypoints2, descriptors2):
    # Example feature matching using FLANN
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.match(descriptors1, descriptors2)
    return matches

def compute_registration_transform(matches, keypoints_ref, keypoints_curr):
    # Example transformation computation using RANSAC
    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    transformation_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return transformation_matrix

# Example function to calculate score based on keypoints and descriptors
def calculate_score(keypoints, descriptors):
    # Example: Calculate score based on the number of keypoints
    return len(keypoints)

# Load orthorectified image paths from a folder
def load_image_paths_from_folder(folder):
    image_paths = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    return image_paths

# Example usage
input_folder = "o_output"
output_folder = "f_output"
image_paths = load_image_paths_from_folder(input_folder)

# Perform feature-based registration with incremental approach
registered_images = incremental_registration(image_paths, output_folder)

# Now you have a list of registered images ready for further processing
