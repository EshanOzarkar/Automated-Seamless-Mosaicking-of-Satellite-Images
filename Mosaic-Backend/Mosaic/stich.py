import cv2
import os
import numpy as np
import json
from tqdm import tqdm

MATCH_SCORES_DIR = '/home/eshan_BE/match_results'

def load_images(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    images = []
    for file in tqdm(sorted(image_files), desc="Loading images"):
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
    return images

def determine_order(match_scores):
    num_images = len(match_scores)
    start_index = np.argmax(np.sum(match_scores, axis=0))
    order = [start_index]
    visited = set(order)
    for _ in range(1, num_images):
        last_index = order[-1]
        next_index = np.argmax(match_scores[last_index])
        while next_index in visited:
            match_scores[last_index][next_index] = 0
            next_index = np.argmax(match_scores[last_index])
        order.append(next_index)
        visited.add(next_index)
    return order

def stitch_images(images):
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, stitched_image = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        print(f"Stitching failed with error code: {status}")
        return None

def create_image_mosaic(folder_path, match_scores_file, OUTPUT_DIR):
    images = load_images(folder_path)
    with open(match_scores_file, 'r') as f:
        match_scores = np.array(json.load(f))
    ordered_indices = determine_order(match_scores)
    ordered_images = [images[i] for i in ordered_indices]
    stitched_image = stitch_images(ordered_images)
    if stitched_image is not None:
        output_dir = os.path.join(OUTPUT_DIR)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        stitched_image_path = os.path.join(output_dir, 'stitched_image.jpg')
        cv2.imwrite(stitched_image_path, stitched_image)
        print(f"Mosaic image saved to {stitched_image_path}")
        return output_dir
    else:
        print("Stitching failed.")
        return None
