import cv2
import numpy as np
import os
from tqdm import tqdm
import json

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def find_matches(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)

def load_images(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if is_image_file(f)]
    images = []
    for file in tqdm(sorted(image_files), desc="Loading images"):
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
    return images

def compute_match_scores(images, output_path):
    num_images = len(images)
    match_scores = np.zeros((num_images, num_images))
    for i in tqdm(range(num_images), desc="Calculating matches"):
        for j in range(i + 1, num_images):
            matches = find_matches(images[i], images[j])
            match_scores[i][j] = match_scores[j][i] = len(matches)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    match_file = os.path.join(output_path, 'match_scores.json')
    with open(match_file, 'w') as f:
        json.dump(match_scores.tolist(), f)

def process_image_matches(folder_path, output_dir):
    images = load_images(folder_path)
    compute_match_scores(images, output_dir)
    return output_dir
