from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time

import radio
import ortho
import match
import stich

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_files():
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded because type difference'},400)

    uploaded_files = request.files.getlist('files[]')

    if not uploaded_files:
        return jsonify({'error': 'No files uploaded'}), 400

    folder_path = os.path.join('uploads', str(int(time.time())))
    os.makedirs(folder_path, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(folder_path, file.filename)
        file_path = file_path.replace('MI1/','')
        file_path = file_path.replace('MI2/','')
        file_path = file_path.replace('MI1\\','')
        file_path = file_path.replace('MI2\\','')
        file.save(file_path)

    print(folder_path)
    run_image_processing_sequence(folder_path)


    print(f'Folder uploaded: {folder_path}')
    return jsonify({'message': 'Folder uploaded successfully'})

@app.route('/output')
def get_output_photo():
    photo_path = os.path.join('output/s-out','stitched_image.jpg')
    return send_file(photo_path,mimetype='image/jpg')



def delete_and_recreate_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        os.rmdir(path)
        print(f"Deleted directory '{path}'")
    os.mkdir(path)
    print(f"Created new directory '{path}'")

def run_image_processing_sequence(image_folder):
    delete_and_recreate_directory("output\s-out")
    output_folder="output/r-out"
    output_path = radio.process_images_for_radiometric_normalization(image_folder, output_folder)
    print(f"All images processed. Check the normalized images at: {output_path}")

    image_folder=output_folder
    output_folder="output/o-out"
    result_folder = ortho.process_images_for_orthorectification(image_folder, output_folder)
    print("All images orthorectified and saved to:", result_folder)
    delete_and_recreate_directory(image_folder)

    result_path = match.process_image_matches(result_folder, "output/m-out/")
    print("Match results saved to:", result_path)

    output_path = stich.create_image_mosaic(result_folder, "output/m-out/match_scores.json", "output/s-out")
    print("Mosaic output available at:", output_path)
    delete_and_recreate_directory(result_folder)
    delete_and_recreate_directory("output/m-out/")
    
    return output_path

if __name__ == '__main__':
    app.run(port=4500)
