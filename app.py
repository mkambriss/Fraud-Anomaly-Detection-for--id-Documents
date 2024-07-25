from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from fraudDetectionModel import FraudDetectionModel
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
fraudDetectionModel = FraudDetectionModel()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            is_real,ocr = fraudDetectionModel.detect_fraud2(file_path)
            text = ""
            if (is_real == 0):
                text = "Fake"
            else:
                text = "Real"    
            return render_template('index.html', filename=filename , text=text , ocr=ocr)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    
    except Exception as e:
        app.logger.error("Error occurred during file upload", exc_info=True)
        flash('An error occurred while uploading the image')
        return redirect(request.url)

@app.route('/upload/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True, port=8080)



