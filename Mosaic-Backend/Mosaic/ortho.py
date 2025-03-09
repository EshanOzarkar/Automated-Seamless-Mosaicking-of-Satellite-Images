import os
import cv2
import numpy as np

def apply_orthorectification_transformation(image, georeference_data):
    geotransform = georeference_data["geotransform"]
    transformation_matrix = np.array([[geotransform[0], geotransform[1], geotransform[2]],
                                      [geotransform[3], geotransform[4], geotransform[5]]], dtype=np.float32)
    orthorectified_image = cv2.warpAffine(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return orthorectified_image

def orthorectify_image(image, georeference_data):
    orthorectified_image = apply_orthorectification_transformation(image, georeference_data)
    return orthorectified_image

def process_images_for_orthorectification(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".tif", ".jpg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                georeference_data_example = {"geotransform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
                orthorectified_image = orthorectify_image(image, georeference_data_example)
                output_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_image_path, orthorectified_image)
                print("Orthorectification completed for:", filename)
            else:
                print("Failed to load image:", filename)

    return output_folder
