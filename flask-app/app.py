import ctypes

import numpy as np
from PIL import Image
from flask import render_template, request, redirect, url_for, flash, session, Flask
from scipy.stats import stats
from werkzeug.utils import secure_filename
import os
from import_lib import init_lib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

my_lib = init_lib()

MLP_MODEL_PATH = b"./best_MLP_model.json"
mlp_model = my_lib.loads_mlp_model(MLP_MODEL_PATH)

# ML_PAPER_VS_OTHER_MODEL_PATH=[] b"./best_paper_vs_other_model.json"
# ML_METAL_VS_OTHER_MODEL_PATH=[] b"./best_metal_vs_other_model.json"
# ML_PLASTIC_VS_OTHER_MODEL_PATH=[] b"./best_plastic_vs_other_model.json"

# paper_vs_other_model = my_lib.loads_linear_model(ML_PAPER_VS_OTHER_MODEL_PATH)
# metal_vs_other_model = my_lib.loads_linear_model(ML_METAL_VS_OTHER_MODEL_PATH)
# plastic_vs_other_model = my_lib.loads_linear_model(ML_PLASTIC_VS_OTHER_MODEL_PATH)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(file_path, image_size):
    with Image.open(file_path) as img:
        img = img.convert('L')

        img_array = np.array(img).astype(float) / 255.0

        img_array = stats.zscore(img_array.flatten()).reshape(image_size, image_size)
        image_vec_flat = img_array.flatten()

    return image_vec_flat


@app.route('/', methods=['GET', 'POST'])
def index():

    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            _, extension = os.path.splitext(filename)
            new_filename = "image" + extension
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            with Image.open(file) as img:
                img = img.resize((32, 32), Image.LANCZOS)
                img.save(new_file_path)

            session['image'] = new_filename

        selected_action = request.form.get('selectedAction')
        if selected_action == "mlp":
            image_vec = process_image(new_file_path, 32)
            image_vec_p = np.ctypeslib.as_ctypes(np.array(image_vec, dtype=ctypes.c_float))
            res_arr = my_lib.predict_mlp(mlp_model, image_vec_p)
            tab = [res_arr[0], res_arr[1], res_arr[2]]
            max_value = max(tab)
            max_index = tab.index(max_value)
            if max_index == 0:
                result = "Metal"
            elif max_index == 1:
                result = "Paper"
            elif max_index == 2:
                result = "Plastic"
        
        session['result'] = result

        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
