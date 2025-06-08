import os
import secrets
import numpy as np
import pydicom
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Flask Configuration
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit upload size to 16MB

db = SQLAlchemy(app)

# Ensure required folders exist
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

MODEL_PATH = "C:/Users/chinm/OneDrive/Desktop/Project/segnet_liver_tumor.keras"
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(image_array):
    global model
    if model is None:
        raise RuntimeError("❌ Model is not loaded!")
    """
    Runs inference using the Keras model.
    Ensures the input structure matches the model's expected format.
    """
    print(f"✅ Input tensor shape for inference: {image_array.shape}")  # Debugging info
    start_time = time.time() 
    # Match model input structure
    if isinstance(model.input, list):
        inputs = {model.input[0].name: image_array}
    else:
        inputs = {model.input.name: image_array}

    results = model.predict(inputs)  # Perform inference
    inference_time = time.time() - start_time
    print(f"⏱️ Model Inference Time: {inference_time:.2f} seconds")
    return results[0]  # Return the first (and only) batch element

# Validation Functions
def is_valid_dicom(file_path):
    try:
        print(f"🔍 Checking DICOM file: {file_path}")
        with open(file_path, "rb") as f:
            f.seek(128)
            magic = f.read(4)
            print(f"🔍 Magic number found: {magic}")
            if magic != b"DICM":
                print("❌ Not a valid DICOM file (missing DICM magic number).")
                return False
        pydicom.dcmread(file_path, force=True)
        print(f"📂 Valid DICOM file detected.")
        return True
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False
    
ALLOWED_EXTENSIONS = {'.dcm'}

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

with app.app_context():
    db.create_all()

# Helper Functions
def dicom_to_png(file_path, output_path):
    try:
        # Validate file paths
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"❌ DICOM file does not exist: {file_path}")
        if not output_path or not output_path.lower().endswith(('.png', '.jpg')):
            raise ValueError(f"❌ Output path must have a valid image extension (.png or .jpg): {output_path}")

        dicom = pydicom.dcmread(file_path)
        image_data = dicom.pixel_array

        # Normalize and convert to 8-bit image
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        image_data = (image_data * 255).astype(np.uint8)
        image = Image.fromarray(image_data).convert("RGB")

        print(f"💾 Saving image to {output_path}")
        image.save(output_path)

    except Exception as e:
        print(f"❌ Error in dicom_to_png(): {e}")
        raise

def preprocess_dicom(file_path):
    """
    Preprocess DICOM to ensure the output is always in (1, 512, 512, 1) format.
    """
    try:
        dicom = pydicom.dcmread(file_path)

        # Ensure DICOM contains pixel data
        if not hasattr(dicom, "pixel_array"):
            raise ValueError("❌ DICOM file does not contain pixel data!")

        image = dicom.pixel_array.astype(np.float32)

        # Debug: Check if DICOM loaded correctly
        print(f"📂 DICOM file loaded: {file_path}")
        print(f"📏 Image shape before resize: {image.shape}")

        # Handle possible empty image
        if image.size == 0:
            raise ValueError(f"❌ Error: Image is empty or invalid! Shape: {image.shape}")

        # Normalize image
        max_val = np.max(image)
        print(f"🛠 Image max value: {max_val}")
        image = image / max_val if max_val > 0 else image

        # Resize image to (512, 512)
        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        # Expand dimensions to ensure (1, 512, 512, 1)
        processed_image = np.expand_dims(resized_image, axis=-1)  # Add channel dimension
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

        print(f"✅ Preprocessed image shape: {processed_image.shape}")
        return processed_image

    except Exception as e:
        print(f"❌ Error in preprocess_dicom: {e}")
        return None

def process_xray(file_path):
    """
    Preprocess X-ray DICOM file for prediction ensuring the output is (1, 512, 512, 1) format.
    """
    print(f"📂 DICOM file loaded: {file_path}")
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)

    # Validate image
    if image.size == 0:
        raise ValueError(f"❌ Error: Image is empty or invalid! Shape: {image.shape}")

    print(f"📏 Image shape before resize: {image.shape}")
    print(f"🛠 Image max value: {np.max(image)}")

    # Handle different image dimensions
    if len(image.shape) == 4:
        image = image[0, :, :, 0]
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]

    # Ensure numerical stability
    image = np.nan_to_num(image).astype(np.float32)

    # Resize image to (512, 512)
    resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    # Expand dimensions to ensure (1, 512, 512, 1)
    processed_image = np.expand_dims(resized_image, axis=-1)  # Add channel dimension
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    print(f"✅ Image resized successfully! New shape: {processed_image.shape}")
    return processed_image

def postprocess_mask(mask, filename):
    if not filename or '.' not in filename:
        raise ValueError(f"Invalid filename: {filename}")
    # Ensure the mask is squeezed to the correct shape
    mask = np.squeeze(mask)
    
    # Convert the mask to a binary format (0s and 1s)
    mask = (mask > 0.5).astype(np.uint8)

    # Create an empty array for the colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Assign colors: Tumor areas in red, background in grayscale
    colored_mask[mask == 1] = [255, 0, 0]  # Red for tumor areas

    # Save the colored mask image
    mask_image = Image.fromarray(colored_mask)
    output_filename = os.path.splitext(filename)[0] + '_segmentation.png'
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    mask_image.save(output_path)

    return output_filename

def stack_images(original_path, segmented_path, output_filename):
    original = Image.open(original_path).convert("RGB")
    segmented = Image.open(segmented_path).convert("RGB")
    original_np = np.array(original).astype(np.float32) / 255.0
    segmented_np = np.array(segmented).astype(np.float32) / 255.0

    # Blend the images for visualization: keeping original in grayscale and overlaying tumor areas in color
    stacked_np = cv2.addWeighted(original_np, 0.7, segmented_np, 0.3, 0)
    stacked_image = Image.fromarray((stacked_np * 255).astype(np.uint8))
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    stacked_image.save(output_path)
    return output_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('❌ Username already exists!', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('❌ Email already exists!', 'danger')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('✅ Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('❌ Invalid credentials', 'danger')
            return redirect(url_for('login'))
        session['logged_in'] = True
        session['user_id'] = user.id
        flash('✅ Login successful!', 'success')
        return redirect(url_for('upload'))
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('❌ No account associated with this email.', 'danger')
            return redirect(url_for('forgot_password'))

        flash('✅ Password reset email sent. Please check your inbox.', 'success')
        return redirect(url_for('reset_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"success": False, "message": "❌ No account associated with this email."})

        user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.session.commit()
        return jsonify({"success": True, "message": "✅ Password successfully changed!"})

    return render_template('reset_password.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('✅ You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        flash('❌ Please log in first!', 'warning')
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('❌ No file uploaded.', 'danger')
            return redirect(url_for('upload'))
        file = request.files['file']
        if file.filename == '':
            flash('❌ No file selected.', 'danger')
            return redirect(url_for('upload'))
        if not file.filename.endswith('.dcm'):
            flash('❌ Only .dcm files are allowed!', 'danger')
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Preprocess & Predict Segmentation
        try:
            preprocessed_image = process_xray(file_path)
            start_time = time.time()
            predicted_mask = predict(preprocessed_image)
            duration = round(time.time() - start_time, 3)  # In seconds
            print(f"✅ Model Inference Time: {duration}s")
            predicted_mask = np.array(predicted_mask)
        except Exception as e:
            print(f'❌ Prediction error: {e}')
            flash('❌ Processing failed.', 'danger')
            return redirect(url_for('upload'))

        if preprocessed_image is None:
            flash('❌ Processing failed.', 'danger')
            return redirect(url_for('upload'))

        segmented_filename = postprocess_mask(predicted_mask, filename)
        original_filename = os.path.splitext(filename)[0] + '_original.png'
        original_path = os.path.join(PROCESSED_FOLDER, original_filename)
        dicom_to_png(file_path, original_path)
        stacked_filename = stack_images(original_path, os.path.join(PROCESSED_FOLDER, segmented_filename), 
                                        os.path.splitext(filename)[0] + '_stacked.png')
        flash('✅ DICOM file processed successfully!', 'success')

        return redirect(url_for('result', stacked_filename=stacked_filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/result')
def result():
    stacked_filename = request.args.get('stacked_filename')
    if not stacked_filename:
        flash('❌ No file found for processing!', 'danger')
        return redirect(url_for('upload'))
    stacked_path = url_for('processed_file', filename=stacked_filename)
    return render_template('result.html', stacked_image_url=stacked_path)

# Run App
if __name__ == '__main__':
    app.run(debug=True)