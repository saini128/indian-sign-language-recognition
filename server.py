from flask import Flask, render_template, Response
import cv2
import main
from tensorflow import keras
from keras.models import load_model
import joblib

app = Flask(__name__)

# Function to capture video frames
model = load_model('hand_model.h5')
label_encoder = joblib.load('label_encoder.joblib')
def generate_frames(model, label_encoder):
    cap = cv2.VideoCapture(0)

    for frame in main.test_model(cap, model, label_encoder):
        yield frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    model = load_model('hand_model.h5')
    label_encoder = joblib.load('label_encoder.joblib')
    return Response(generate_frames(model, label_encoder), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)