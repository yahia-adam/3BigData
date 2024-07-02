from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            session['image'] = filename

        selected_action = request.form.get('selectedAction')
        session['result'] = f"Processed with {selected_action}"

        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
