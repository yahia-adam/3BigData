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

ML_PAPER_VS_OTHER_MODEL_PATH = b"./models/best_paper_vs_other.json"
ML_METAL_VS_OTHER_MODEL_PATH = b"./models/best_metal_vs_other.json"
ML_PLASTIC_VS_OTHER_MODEL_PATH = b"./models/best_plastic_vs_other.json"

paper_vs_other_model = my_lib.loads_linear_model(ML_PAPER_VS_OTHER_MODEL_PATH)
metal_vs_other_model = my_lib.loads_linear_model(ML_METAL_VS_OTHER_MODEL_PATH)
plastic_vs_other_model = my_lib.loads_linear_model(ML_PLASTIC_VS_OTHER_MODEL_PATH)

MLP_MODEL_PATH = b"./models/best_mlp_model.json"
mlp_model = my_lib.loads_mlp_model(MLP_MODEL_PATH)


RBF_PAPER_VS_OTHER_MODEL_PATH = b"./models/rbf_best_paper_vs_other.json"
RBF_METAL_VS_OTHER_MODEL_PATH = b"./models/rbf_best_metal_vs_other.json"
RBF_PLASTIC_VS_OTHER_MODEL_PATH = b"./models/rbf_best_plastic_vs_other.json"

rbf_paper_vs_other_model = my_lib.load_rbf_model(RBF_PAPER_VS_OTHER_MODEL_PATH)
rbf_metal_vs_other_model = my_lib.load_rbf_model(RBF_METAL_VS_OTHER_MODEL_PATH)
rbf_plastic_vs_other_model = my_lib.load_rbf_model(RBF_PLASTIC_VS_OTHER_MODEL_PATH)

paper_vs_other_model = my_lib.loads_linear_model(ML_PAPER_VS_OTHER_MODEL_PATH)
metal_vs_other_model = my_lib.loads_linear_model(ML_METAL_VS_OTHER_MODEL_PATH)
plastic_vs_other_model = my_lib.loads_linear_model(ML_PLASTIC_VS_OTHER_MODEL_PATH)

MLP_MODEL_PATH = b"./models/best_mlp_model.json"
mlp_model = my_lib.loads_mlp_model(MLP_MODEL_PATH)

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

        image_vec = process_image(new_file_path, 32)
        image_vec_p = np.ctypeslib.as_ctypes(np.array(image_vec, dtype=ctypes.c_float))

        if selected_action == "mlp":
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

        if selected_action == "lm":
            metal_predict = my_lib.predict_linear_model(metal_vs_other_model, image_vec_p)
            paper_predict = my_lib.predict_linear_model(paper_vs_other_model, image_vec_p)
            plastic_predict = my_lib.predict_linear_model(plastic_vs_other_model, image_vec_p)

            if (metal_predict >= paper_predict and metal_predict >= plastic_predict):
                result = "Metal"
            elif (paper_predict >= metal_predict and paper_predict >= plastic_predict):
                result = "Paper"
            elif (plastic_predict >= metal_predict and plastic_predict >= paper_predict):
                result = "Plastic"
        if selected_action == "rbf":
            metal_predict = my_lib.predict_rbf_classification(metal_vs_other_model, image_vec_p)
            paper_predict = my_lib.predict_rbf_classification(paper_vs_other_model, image_vec_p)
            plastic_predict = my_lib.predict_rbf_classification(plastic_vs_other_model, image_vec_p)

            if (metal_predict > paper_predict and metal_predict > plastic_predict):
                result = "Metal"
            elif (paper_predict > metal_predict and paper_predict > plastic_predict):
                result = "Paper"
            elif (plastic_predict > metal_predict and plastic_predict > paper_predict):
                result = "Plastic"


        session['result'] = result

        return redirect(url_for('index'))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)