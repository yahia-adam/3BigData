from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
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
                img = img.resize((32, 32))
                img.save(new_file_path)

            session['image'] = new_filename

        selected_action = request.form.get('selectedAction')
        session['result'] = "A voir"

        return redirect(url_for('index'))

    return render_template('index.html')


# @app.route('/test_cases', methods=['GET', 'POST'])
# def test_cases():
#     models = ['Linear Model', 'Radial Basis Function', 'Support Vector Machine', 'Multi Layer Perceptron']
#     test_cases = ['Linear Simple', 'Linear Multiple', 'Multi Linear 3 Classes', 'XOR', 'Cross', 'Multi Cross',
#                   'Linear Simple 2D', 'Non Linear Simple 2D', 'Linear Simple 3D', 'Linear Tricky 3D',
#                   'Non Linear Simple 3D']
#     selected_model = None
#     selected_test_case = None
#
#     if request.method == 'POST':
#         selected_model = request.form.get('model')
#         selected_test_case = request.form.get('test_case')
#
#     return render_template('test_cases.html',
#                            models=models,
#                            test_cases=test_cases,
#                            selected_test_case=selected_test_case)


if __name__ == '__main__':
    app.run(debug=True)
