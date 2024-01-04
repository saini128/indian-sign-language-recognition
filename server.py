from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import main
from tensorflow import keras
from keras.models import load_model
import joblib
import numpy as np
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
model = load_model('hand_model.h5')
label_encoder = joblib.load('label_encoder.joblib')
words = []
current_language='en'
translations = {
    'Nameste': {'en': 'Nameste', 'hi': 'नमस्ते', 'pa': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ'},
    'Bye': {'en': 'Bye', 'hi': 'अलविदा', 'pa': 'ਅਲਵਿਦਾ'},
    "Good": {'en': 'Good', 'hi': 'अच्छा', 'pa': 'ਚੰਗਾ'},
    "Morning": {'en': 'Morning', 'hi': 'सुबह', 'pa': 'ਸਵੇਰ'},
    "Afternoon": {'en': 'Afternoon', 'hi': 'दोपहर', 'pa': 'ਦੋਪਹਿਰ'},
    "How are you?": {'en': 'How are you?', 'hi': 'आप कैसे हैं?', 'pa': 'ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?'},
    "Do": {'en': 'Do', 'hi': 'करो', 'pa': 'ਕਰੋ'},
    "Indian": {'en': 'Indian', 'hi': 'भारतीय', 'pa': 'ਭਾਰਤੀ'},
    "Computer": {'en': 'Computer', 'hi': 'कंप्यूटर', 'pa': 'ਕੰਪਿਊਟਰ'},
    "Thank You": {'en': 'Thank You', 'hi': 'धन्यवाद', 'pa': 'ਧੰਨਵਾਦ'},
    "I": {'en': 'I', 'hi': 'मैं', 'pa': 'ਮੈਂ'},
    "What": {'en': 'What', 'hi': 'क्या', 'pa': 'ਕੀ'},
    "You": {'en': 'You', 'hi': 'तुम', 'pa': 'ਤੁਸੀਂ'},
    "Worked Hard": {'en': 'Worked Hard', 'hi': 'मेहनत की', 'pa': 'ਮਿਹਨਤ ਕੀਤੀ'},
    "Today": {'en': 'Today', 'hi': 'आज', 'pa': 'ਅੱਜ'},
    "We": {'en': 'We', 'hi': 'हम', 'pa': 'ਅਸੀਂ'},
}
def process_frames(frames, model, label_encoder):
    frames_array = np.array(frames)
    averaged_landmarks = np.mean(frames_array, axis=0)

    input_features = averaged_landmarks.flatten()

    prediction = model.predict(np.array([input_features]))
    predicted_class = np.argmax(prediction)
    predicted_word = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_word


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('language_change')
def handle_language_change(new_language):
    global current_language
    current_language = new_language

def translate_words(words, language):
    translated_words = [translations[word][language] if word in translations else '' for word in words]
    return translated_words

def generate_frames(model, label_encoder):
    words.clear()
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frames = []
    predicted_word = ""
    previous_word = ""
    translated_words = [] 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

        input_features = np.concatenate([pose, lh, rh])

        frames.append(input_features)

        if len(frames) == 10:
            predicted_word = process_frames(frames, model, label_encoder) #to explain
            frames = []
            if not np.any(lh) and not np.any(rh):
                predicted_word = ""

            if predicted_word != previous_word:
                words.append(',')
                words.append(predicted_word)
                previous_word = predicted_word
                
        translated_words = translate_words(words,current_language) # to explain
        socketio.emit('update_words', ' '.join(translated_words))

        mp_drawing.draw_landmarks(rgb_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.putText(rgb_frame, f"{predicted_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', rgb_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/reset')
def reset():
    words.clear()
    return Response('Success')

@app.route('/video_feed')
def video_feed():
    model = load_model('hand_model.h5')
    label_encoder = joblib.load('label_encoder.joblib')
    return Response(generate_frames(model, label_encoder), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=True, port=3000)
