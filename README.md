# Automated Seamless Mosaicking of Satellite Images

A web application that combines overlapping satellite image tiles into a single mosaic. A Flask backend runs a classical computer-vision pipeline (OpenCV and scikit-image), and a Next.js front end handles upload and display.

## Pipeline

1. **Upload**: the front end posts the image tiles to POST /upload (multipart field files[]).
2. **Radiometric normalisation**: contrast-limited adaptive histogram equalisation (CLAHE) on each colour channel, to reduce brightness and contrast differences between tiles.
3. **Orthorectification**: an affine warp driven by georeference metadata (a simplified approach, not full terrain-based orthorectification).
4. **Pairwise matching**: ORB features with a brute-force Hamming matcher score every pair of tiles; the scores are saved as a JSON matrix.
5. **Ordering**: tiles are ordered greedily, starting from the best-connected tile and following the highest match scores.
6. **Stitching**: OpenCV Stitcher in panorama mode produces the final mosaic.
7. **Result**: GET /output returns stitched_image.jpg.

## Project layout

~~~
Mosaic-Backend/Mosaic/   Flask API (main.py) and pipeline modules: radio.py, ortho.py, match.py, stich.py
Mossaic-Frontend/        Next.js 14 (TypeScript, Tailwind CSS) with Clerk authentication
~~~

## Tech stack

- Backend: Python, Flask, Flask-CORS, OpenCV, scikit-image, NumPy, tqdm
- Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS, Framer Motion, Clerk

## Running locally

Backend:

~~~bash
cd Mosaic-Backend/Mosaic
pip install flask flask-cors opencv-python scikit-image numpy tqdm
python main.py   # serves on http://localhost:4500
~~~

Frontend:

~~~bash
cd Mossaic-Frontend
npm install
npm run dev
~~~

The front end uses Clerk for sign-in. Create a Clerk application and add its keys to .env.local (NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY and CLERK_SECRET_KEY).

## Limitations

This is an academic prototype.

- No requirements.txt and no automated tests
- Some paths are hard-coded, for example in stich.py
- Processing runs synchronously inside the upload request, so large inputs are slow
- Orthorectification is simplified, and the pipeline uses classical computer vision rather than machine learning
- The backend has no authentication of its own

## Possible improvements

- Move processing to a background worker with a task-ID status endpoint
- Pin dependencies and add tests
- Replace hard-coded paths with configuration
# Mosaic
