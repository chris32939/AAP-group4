import os
from flask import Flask, jsonify, request, render_template, redirect, url_for, session, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from flask_socketio import SocketIO
from flask_cors import CORS
import tensorflow as tf
from PIL import Image, ImageDraw
# Ensure ffmpeg is installed 'conda install -c conda-forge ffmpeg'. ffmpeg is used in a subprocess.

import cv2
import numpy as np
import torch
import time
import datetime
import secrets
import warnings
import subprocess

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Secret key for session management
app.secret_key = os.urandom(24)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking
db = SQLAlchemy(app)

# # Configure Flask-Mail to use MailHog SMTP server
    # app.config['MAIL_SERVER'] = 'localhost'  # MailHog SMTP server
    # app.config['MAIL_PORT'] = 1025  # MailHog default SMTP port
    # app.config['MAIL_USE_TLS'] = False  # MailHog does not require TLS
    # app.config['MAIL_USE_SSL'] = False  # MailHog does not require SSL
    # app.config['MAIL_USERNAME'] = ''  # MailHog does not require a username
    # app.config['MAIL_PASSWORD'] = ''  # MailHog does not require a password
    # app.config['MAIL_DEFAULT_SENDER'] = 'noreply@example.com'  # Default sender for outgoing mail
    # mail = Mail(app)

# Configure Flask-Mail to use Gmail SMTP server
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Gmail SMTP server
app.config['MAIL_PORT'] = 587  # Gmail SMTP port
app.config['MAIL_USE_TLS'] = True  # Enable TLS encryption
app.config['MAIL_USE_SSL'] = False  # No SSL needed, since we are using TLS
app.config['MAIL_USERNAME'] = 'c2729751@gmail.com'  # Your Gmail address
app.config['MAIL_PASSWORD'] = 'tgmm zttp dggf iepo'  # Your Gmail app-specific password
app.config['MAIL_DEFAULT_SENDER'] = 'c2729751@gmail.com'  # Default sender
mail = Mail(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Initialize a serializer for generating secure tokens
s = URLSafeTimedSerializer(app.secret_key)

# Initialize the model
object_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/christopher.pt')

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")


########## Define Classes ##########
# Define User model for the SQLite database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=False, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    reset_token = db.Column(db.String(200), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)

# Cross-site request forgery protection
class LoginForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

class SignupForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])

class ChangeUserForm(FlaskForm):
    new_username = StringField('New Username', validators=[DataRequired(), Length(min=3, max=20)])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])

class RequestResetPasswordForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])

class ResetPasswordForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])

class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_url = db.Column(db.String(200), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)
    user = db.relationship('User', backref=db.backref('files', lazy=True))

    def __init__(self, user_id, file_url, file_type, timestamp):
        self.user_id = user_id
        self.file_url = file_url
        self.file_type = file_type
        self.timestamp = timestamp

# Create the database
with app.app_context():
    db.create_all()

# Add default user to the database if it doesn't exist
def add_default_user():
    user = User.query.filter_by(email='c2729751@gmail.com').first()
    if not user:
        hashed_password = generate_password_hash("password123")
        new_user = User(email="c2729751@gmail.com", username="admin", password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        print("Default user 'admin' added to the database.")

# Run the add_default_user function once when starting the app
with app.app_context():
    add_default_user()

# User loader for LoginManager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


########## Setup routes ##########
    ##### Christopher's routes #####
@app.route("/", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if request.method == "POST":
        if form.validate_on_submit():
            email = form.email.data
            password = form.password.data
            user = User.query.filter_by(email=email).first()

            if user and check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for("index"))
            else:
                return render_template("login.html", form=form, error="Invalid email or password")
    
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()

    session.clear()

    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()

    if request.method == "POST":
        email = form.email.data
        username = form.username.data
        password = form.password.data
        confirm_password = form.confirm_password.data
        
        # Check if the email already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template("signup.html", form=form, error="Email already exists")

        # Check if passwords match
        if password != confirm_password:
            return render_template("signup.html", form=form, error="Passwords do not match.")

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)
        
        # Create a new user and add to the database
        new_user = User(email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # Redirect to login page after successful sign-up
        return redirect(url_for("login"))

    return render_template("signup.html", form=form)

@app.route("/changeUserDetails", methods=["GET", "POST"])
@login_required
def change_user_details():
    form = ChangeUserForm()

    if request.method == "POST":
        # Retrieve the values from the form
        new_username = form.new_username.data
        new_password = form.new_password.data
        confirm_password = form.confirm_password.data

        # Check if password fields are empty
        if not new_password or not confirm_password:
            return render_template("changeUserDetails.html", form=form, error="Password fields cannot be empty.")

        # Check if passwords match
        if new_password != confirm_password:
            return render_template("changeUserDetails.html", form=form, error="Passwords do not match.")

        # Update user details in the database
        current_user.username = new_username
        current_user.password = generate_password_hash(new_password)
        db.session.commit()

        # Return success message
        return render_template("changeUserDetails.html", form=form, success="User details updated successfully.")

    # If GET request, render the change user details page with the form
    return render_template("changeUserDetails.html", form=form)

@app.route("/request_reset_password", methods=["GET", "POST"])
def request_reset_password():
    form = RequestResetPasswordForm()

    if request.method == "POST":
        email = request.form.get("email")
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate a token with the email
            token = s.dumps(email, salt='password-reset-salt')
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Send reset email
            msg = Message("Password Reset Request", recipients=[email])
            msg.body = f"Click the link to reset your password: {reset_url}"
            try:
                mail.send(msg)

                # Return success message
                return render_template("requestResetPassword.html", form=form, success="Reset email has been sent.")
            except Exception as e:
                return render_template("requestResetPassword.html", form=form, error="Email not found.")
        
        return render_template("requestResetPassword.html", form=form, error="Email not found.")
    return render_template("requestResetPassword.html", form=form)

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    form = ResetPasswordForm()

    try:
        email = s.loads(token, salt='password-reset-salt', max_age=300)
        print(email)
        user = User.query.filter_by(email=email).first()
        
        if request.method == "POST":
            new_password = request.form.get("password")
            if user:
                user.password = generate_password_hash(new_password)
                user.reset_token = None
                user.reset_token_expiry = None
                db.session.commit()
                return redirect(url_for('login'))
            
            return render_template("login.html", form=form, error="User not found.")
    except Exception as e:
        return render_template("login.html", form=form, error="The password reset link is invalid or expired.")

    return render_template("resetPassword.html", form=form, token=token)

@app.route("/index")
@login_required
def index():
    return render_template('index.html', username=current_user.username)

@app.route("/objectDetection")
@login_required
def objectDetection():
    # Retrieve all files for the current user
    user_files = DetectionHistory.query.filter_by(user_id=current_user.id).all()

    file_data = [{
        'file_url': file.file_url,
        'file_type': file.file_type,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file.timestamp))
    } for file in user_files]

    return render_template('objectDetection.html', files=file_data, username=current_user.username)

@app.route("/runObjectDetection", methods=['POST'])
@login_required
def runObjectDetection():
    # Load file from POST request
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected."})

    # Check file extension to determine if it's an image or video
    file_extension = uploaded_file.filename.split('.')[-1].lower()

    # If it's an image (jpg, jpeg, png, etc.)
    if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
        # Process image
        imgbytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(imgbytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Image could not be decoded."})

        # Resize the image to 640x640 (YOLOv5 input size)
        img = cv2.resize(img, (640, 640))

        # Process the image for YOLOv5
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (YOLOv5 expects RGB images)
        
        # Run detection with YOLOv5 model
        results = object_detection_model(img_rgb)  # Detect on the image

        # Get detections as pandas DataFrame
        predictions = results.pandas().xywh[0].to_dict(orient="records")  # Convert DataFrame to a list of dicts
        
        # List of colours for bounding boxes
        colours = {
            'chair': (255, 255, 0),
            'plant': (255, 165, 0),
            'sofa': (0, 255, 255),
            'table': (0, 255, 0)
        }

        # Draw bounding boxes and labels on the image
        detection_results = []  # To store detection data for JSON response
        for i, pred in enumerate(predictions):
            # Extract the center and width/height for the bounding box
            xcenter, ycenter, width, height = pred['xcenter'], pred['ycenter'], pred['width'], pred['height']
            
            # Calculate the corner points (xmin, ymin, xmax, ymax) directly
            xmin = int(xcenter - width / 2)
            xmax = int(xcenter + width / 2)
            ymin = int(ycenter - height / 2)
            ymax = int(ycenter + height / 2)

            # Get the colour for the bounding box (cycle through the colours list)
            object_name = pred['name'].lower()  # Make the name lowercase to match the dictionary keys
            colour = colours.get(object_name, (255, 255, 255))

            # Draw the bounding box on the image
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colour, 2)

            # Add label with confidence score
            label = f"{pred['name']} {pred['confidence']:.2f}"
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Add detection result to the list for JSON response
            detection_results.append({
                "class": pred['name'],
                "confidence": pred['confidence'],
                "bounding_box": [xmin, ymin, xmax, ymax]
            })

        # Save the processed image as a PNG in the static folder
        timestamp = str(int(time.time()))  # Use a timestamp to make the filename unique
        output_image_path = os.path.join('static', f'output_image_{timestamp}.png')
        cv2.imwrite(output_image_path, img)  # Save image as PNG

        # Create a new entry in the ImageHistory table for the logged-in user
        new_file = DetectionHistory(user_id=current_user.id, file_url=f"/static/{os.path.basename(output_image_path)}", file_type="image", timestamp=int(time.time()))
        db.session.add(new_file)
        db.session.commit()

        # Return the image URL in the response
        file_url = f"/static/{os.path.basename(output_image_path)}"
        
        # Respond with the image URL and detections
        response = {
            "file_type": "image",
            "output_url": f"/static/{os.path.basename(file_url)}",  # The URL to access the saved image
            "detections": detection_results  # The JSON object containing class labels, confidence, and bounding box coordinates
        }
        return jsonify(response)

    # If it's a video (mp4, avi, etc.)
    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        # Suppress specific FutureWarnings
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

        # Save the video file temporarily to process
        video_path = os.path.join('static', f"{secrets.token_hex(8)}.{file_extension}")
        print(f"Saving video file as {video_path}")
        uploaded_file.save(video_path)
        
        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file."})

        # Define codec and output video file (to save the processed video)
        output_video_path = os.path.join('static', f"output_video_{int(time.time())}.mp4")
        print(f"Opening video writer for {output_video_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 360))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0

        # List to store detection results for each frame
        video_detections = []   

        # Process each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Update the progress
            frame_counter += 1
            progress = (frame_counter / total_frames) * 100  # Calculate percentage
            socketio.emit('progress', {'progress': progress})  # Send progress to frontend

            # Resize frame to (640, 640) for detection input
            frame_resized  = cv2.resize(frame, (640, 360))

            # Resize the frame to 640x640 (YOLOv5 input size)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (YOLOv5 expects RGB images)
            
            # Run detection with YOLOv5 model
            results = object_detection_model(img_rgb)  # Detect on the frame

            # Get detections as pandas DataFrame
            predictions = results.pandas().xywh[0].to_dict(orient="records")  # Convert DataFrame to a list of dicts

            # List of colours for bounding boxes
            colours = {
                'chair': (255, 255, 0),
                'plant': (255, 165, 0),
                'sofa': (0, 255, 255),
                'table': (0, 255, 0)
            }
            
            # Draw bounding boxes and labels on the frame
            frame_detections = []  # To store detections for the current frame
            for pred in predictions:
                # Extract the center and width/height for the bounding box
                xcenter, ycenter, width, height = pred['xcenter'], pred['ycenter'], pred['width'], pred['height']
                
                # Calculate the corner points (xmin, ymin, xmax, ymax) directly
                xmin = int(xcenter - width / 2)
                xmax = int(xcenter + width / 2)
                ymin = int(ycenter - height / 2)
                ymax = int(ycenter + height / 2)

                # Get the colour for the bounding box (cycle through the colours list)
                object_name = pred['name'].lower()  # Make the name lowercase to match the dictionary keys
                colour = colours.get(object_name, (255, 255, 255))

                # Draw the bounding box on the frame
                cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), colour, 2)

                # Add label with confidence score
                label = f"{pred['name']} {pred['confidence']:.2f}"
                cv2.putText(frame_resized, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

                # Add detection result to the list for the current frame
                frame_detections.append({
                    "class": pred['name'],
                    "confidence": pred['confidence'],
                    "bounding_box": [xmin, ymin, xmax, ymax]
                })

            # Add the detections for this frame to the overall video detections list
            video_detections.append(frame_detections)

            # Write the processed frame to the output video
            out.write(frame_resized)

        # Release the video capture and writer objects
        cap.release()
        print(f"Releasing video writer for {output_video_path}")
        out.release()

        # Create output video file
        final_output_path = os.path.join(os.getcwd(), 'static', f"output_video_{int(time.time())}.mp4")
        output_video_dir = os.path.join(os.getcwd(), output_video_path)

        # Ensure ffmpeg is installed 'conda install -c conda-forge ffmpeg'
        subprocess.call(['ffmpeg', '-i', output_video_dir, final_output_path])

        # Create a new entry in the ImageHistory table for the logged-in user
        new_file = DetectionHistory(user_id=current_user.id, file_url=f"/static/{os.path.basename(final_output_path)}", file_type="video", timestamp=int(time.time()))
        db.session.add(new_file)
        db.session.commit()

        # Return the URL of the processed video and detection results for each frame
        processed_video_url = f"/static/{os.path.basename(final_output_path)}"

        return jsonify({
            "file_type": "video",
            "output_url": processed_video_url,  # The URL to access the processed video
            "detections": video_detections  # The list of detection results for each frame
        })

    else:
        return jsonify({"error": "Unsupported file type."})

@app.route("/objectDetectionHistory")
@login_required
def objectDetectionHistory():
    # Retrieve all images for the current user
    user_files = DetectionHistory.query.filter_by(user_id=current_user.id).all()    

    file_data = [{
        'file_url': file.file_url,
        'file_type': file.file_type,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file.timestamp))
    } for file in user_files]

    # Return JSON response containing image history
    return jsonify({'files': file_data})   

@app.route("/deleteObjectDetectionHistory", methods=["POST"])
@login_required
def deleteObjectDetectionHistory():
    # Query the ImageHistory table to find all images of the logged-in user
    files_to_delete = DetectionHistory.query.filter_by(user_id=current_user.id).all()

    if not files_to_delete:
        return jsonify({"error": "No files found to delete."}), 400

    # Delete all the images associated with the logged-in user
    for file in files_to_delete:
        # If you want to remove the image from the file system as well, you can use `os.remove(image.image_url)`
        if os.path.exists(file.file_url[1:]):  # Removing the leading '/' from the path
            os.remove(file.file_url[1:])  # Delete the image file from the server
        
        db.session.delete(file)  # Delete the image entry from the database
    
    db.session.commit()

    # Return a success response
    return jsonify({"message": "Detection history deleted successfully."}), 200

@app.route('/saveBoxCoordinates', methods=['POST'])
@login_required
def save_coordinates():
    bounding_box_coordinates = request.json.get('boundingBoxCoordinates')

    # Store the coordinates in the session
    session['boundingBoxCoordinates'] = bounding_box_coordinates

    return jsonify({"message": "Coordinates saved to session"}), 200

@app.route('/getBoxCoordinates', methods=['GET'])
@login_required
def get_coordinates():
    # Get the bounding box coordinates from the session
    bounding_box_coordinates = session.get('boundingBoxCoordinates', None)

    if bounding_box_coordinates:    
        return jsonify({"boundingBoxCoordinates": bounding_box_coordinates}), 200
    else:
        return jsonify({"message": "No coordinates found in session"}), 404

##### End of Christopher's routes #####

# RenJun #
@app.route('/predictBox', methods=['POST'])
@login_required
def predictBox():
    try:
        box_model = tf.keras.models.load_model('weights/box_position_classifier.h5')
        # Class labels for predictions
        class_mapping = {
            0: "Top-left", 1: "Top-middle", 2: "Top-right",
            3: "Middle-left", 4: "Middle-middle", 5: "Middle-right",
            6: "Bottom-left", 7: "Bottom-middle", 8: "Bottom-right"
        }
        data = request.json
        width, height = data["width"], data["height"]
        xmin, ymin, xmax, ymax = data["xmin"], data["ymin"], data["xmax"], data["ymax"]

        input_data = np.array([[width, height, xmin, ymin, xmax, ymax]], dtype=np.float32)
        prediction = box_model.predict(input_data)
        
        predicted_label = np.argmax(prediction, axis=1)[0]
        readable_label = class_mapping[predicted_label]

        return jsonify({"prediction": readable_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/draw-box', methods=['POST'])
@login_required
def draw_bounding_box():
    try:
        OUTPUT_FOLDER = "static"
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        data = request.json
        width, height = int(data["width"]), int(data["height"])
        xmin, ymin, xmax, ymax = int(data["xmin"]), int(data["ymin"]), int(data["xmax"]), int(data["ymax"])

        # Create an image with a bounding box
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        output_path = os.path.join(OUTPUT_FOLDER, "bounding_box.png")
        image.save(output_path)

        return jsonify({"image_url": f"/static/bounding_box.png"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/box_class")
@login_required
def box_class():
    
    return render_template('box_classifier.html', username=current_user.username)


# Start the web server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
