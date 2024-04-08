import os
import cv2
import numpy as np

# Function for feature extraction using SIFT
def extract_features(image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors

# Function for feature-based registration using SIFT and FlannBasedMatcher
def feature_based_registration(images):
    registered_images = [images[0]]  # Initialize with the first image
    for i in range(1, len(images)):
        # Extract features from the images using SIFT
        keypoints1, descriptors1 = extract_features(images[i - 1])
        keypoints2, descriptors2 = extract_features(images[i])

        # Check keypoints and descriptors before matching
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            print("Warning: No keypoints detected! Skipping registration for this pair.")
            continue  # Skip to the next image pair

        if descriptors1 is None or descriptors2 is None:
            print("Warning: No descriptors detected! Skipping registration for this pair.")
            continue  # Skip to the next image pair

        print("Shape of descriptors1:", descriptors1.shape)
        print("Shape of descriptors2:", descriptors2.shape)

        if descriptors1.size == 0 or descriptors2.size == 0:
            print("Warning: Empty descriptors detected! Skipping registration for this pair.")
            continue  # Skip to the next image pair

        # Initialize FlannBasedMatcher
        flann = cv2.FlannBasedMatcher()
        
        # Match descriptors using KNN
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to select good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Check if enough good matches are found
        if len(good_matches) < 4:
            print("Warning: Insufficient good matches! Skipping registration for this pair.")
            continue  # Skip to the next image pair

        # Estimate the transformation matrix using RANSAC
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        transformation_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply the transformation to register the image
        registered_image = cv2.warpPerspective(images[i], transformation_matrix, (images[i].shape[1], images[i].shape[0]))
        registered_images.append(registered_image)
    return registered_images


# Function to read images from a folder
def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Could not read image: {image_path}")
    return images

# Example input and output folder paths
input_folder = "o_out"
output_folder = "f_out"

# Read images from the input folder
example_images = read_images_from_folder(input_folder)

# Perform feature-based registration
registered_images_example = feature_based_registration(example_images)

# Save the registered images
for i, registered_image in enumerate(registered_images_example):
    output_image_path_registered = os.path.join(output_folder, f"registered_image_{i}.tif")
    cv2.imwrite(output_image_path_registered, registered_image)
    print(f"Feature-Based Registration completed. Registered image {i} is saved at:", output_image_path_registered)
