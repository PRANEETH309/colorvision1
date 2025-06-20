from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import webcolors
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static"

# Ensure static directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Prepare color mapping from HEX to name
hex_to_name = {v: k for k, v in webcolors.CSS3_NAMES_TO_HEX.items()}

def closest_color(requested_color):
    min_dist = float('inf')
    closest_name = None
    for hex_code, name in hex_to_name.items():
        r, g, b = webcolors.hex_to_rgb(hex_code)
        dist = (r - requested_color[0])**2 + (g - requested_color[1])**2 + (b - requested_color[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def analyze_colors(image, num_colors):
    image = cv2.resize(image, (400, 400))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped = image_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(reshaped)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_.reshape((400, 400))

    output = image.copy()
    legend = []

    for i in range(num_colors):
        mask = (labels == i)
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            continue
        cx, cy = int(np.mean(x)), int(np.mean(y))
        rgb = tuple(centers[i])
        name = closest_color(rgb)
        cv2.putText(output, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        legend.append(f"{i + 1}. {name} - RGB{rgb}")

    output_path = os.path.join(UPLOAD_FOLDER, 'labeled_output.jpg')
    cv2.imwrite(output_path, output)

    return legend

@app.route('/', methods=['GET', 'POST'])
def index():
    legend = None
    output_img = None
    num_colors = 5

    if request.method == 'POST':
        try:
            num_colors = int(request.form.get('num_colors', 5))
        except ValueError:
            num_colors = 5

        if 'captured_image' in request.form and request.form['captured_image']:
            data_url = request.form['captured_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)

            img = Image.open(BytesIO(binary_data)).convert('RGB')
            img_np = np.array(img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            legend = analyze_colors(img_cv, num_colors)
            output_img = 'static/labeled_output.jpg'

        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)

                # Ensure static/ exists
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                file.save(path)
                image = cv2.imread(path)
                legend = analyze_colors(image, num_colors)
                output_img = 'static/labeled_output.jpg'

    return render_template('index.html', legend=legend, output_img=output_img, num_colors=num_colors)

if __name__ == '__main__':
    app.run(debug=True)
