from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import cv2
from keras.models import load_model
import subprocess
import webbrowser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

# MongoDB Setup
# Securely fetch URI from environment variables (No hardcoded passwords in code)
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/emoify')
client = MongoClient(MONGO_URI)
db = client.emoify
users_col = db.users

# Load Emotion Detection Model
try:
    model = load_model('model.h5')
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    label_map = ['anger', 'neutral', 'fear', 'happy', 'sad', 'surprise']
except Exception as e:
    print(f"Model loading error: {e}")

info = {}

# -------------------- ROUTES --------------------

@app.route('/')
def home():
    return redirect(url_for('login'))

# User Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        role = request.form.get('role', 'user')

        if role == 'trainer':
            flash('Trainer accounts cannot be created via signup.', 'danger')
            return redirect(url_for('signup'))

        if users_col.find_one({'email': email}):
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        users_col.insert_one({
            'email': email,
            'password': hashed_pw,
            'verified': True
        })

        flash('User account created. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']

        user = users_col.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session.clear()
            session['user'] = email
            return redirect(url_for('index'))
        else:
            flash('Invalid user credentials.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Trainer Login (separate route)
@app.route('/trainer-login', methods=['POST'])
def trainer_login():
    username = request.form['username']
    password = request.form['password']

    if username == 'admin' and password == 'admin':
        session.clear()
        session['trainer'] = username
        return redirect(url_for('trainer'))
    else:
        flash('Username or password is wrong.', 'danger')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/trainer')
def trainer():
    if 'trainer' not in session:
        flash('Please log in as trainer to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('trainer.html', message='')


# -------------------- EMOTION & MUSIC --------------------

@app.route('/choose_singer', methods=['POST'])
def choose_singer():
    info['language'] = request.form['language']
    return render_template('choose_singer.html', data=info['language'])

@app.route('/emotion_detect', methods=['POST'])
def emotion_detect():
    info['singer'] = request.form['singer']
    cap = cv2.VideoCapture(0)
    found = False
    roi = None
    while not found:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                roi = frame[y1:y2, x1:x2]
                cv2.imwrite("static/face.jpg", roi)
                found = True
                break

    cap.release()

    if roi is None:
        flash("Face not detected. Please try again.", "danger")
        return redirect(url_for('index'))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(gray, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

    prediction = model.predict(roi_reshaped)
    emotion = label_map[np.argmax(prediction)]

    if emotion in ["sad", "anger"]:
        return render_template("ask_music_preference.html", data=emotion)

    link = f"https://www.youtube.com/results?search_query={info['singer']}+{emotion}+{info['language']}+song"
    webbrowser.open(link)
    return render_template("emotion_detect.html", data=emotion, link=link)

@app.route('/recommend_music', methods=['POST'])
def recommend_music():
    preference = request.form['preference']
    link = f"https://www.youtube.com/results?search_query={info['singer']}+{preference}+{info['language']}+song"
    webbrowser.open(link)
    return render_template("music_recommendation.html", data=preference, link=link)

# -------------------- TRAINER TOOLS --------------------

@app.route('/collect_data', methods=['POST'])
def collect_data():
    if 'trainer' not in session:
        flash('Please login as trainer to access this.', 'warning')
        return redirect(url_for('login'))

    emotion = request.form['emotion'].strip().lower()
    result = subprocess.run(['python', 'data_collection.py', emotion], capture_output=True, text=True)
    message = f"Data collected for '{emotion}':\n{result.stdout}"
    return render_template('trainer.html', message=message)

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'trainer' not in session:
        flash('Please login as trainer to access this.', 'warning')
        return redirect(url_for('login'))

    result = subprocess.run(['python', 'data_train.py'], capture_output=True, text=True)
    message = f"Model training completed:\n{result.stdout}"
    return render_template('trainer.html', message=message)

# -------------------- RUN --------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
