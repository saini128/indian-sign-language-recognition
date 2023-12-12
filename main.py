import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
import joblib

def form_sentence(recognized_words):
    sentence = ' '.join(recognized_words)
    return sentence

def process_frames(frames, model, label_encoder):
    
    frames_array = np.array(frames)
    averaged_landmarks = np.mean(frames_array, axis=0)

    input_features = averaged_landmarks.flatten()

    prediction = model.predict(np.array([input_features]))
    predicted_class = np.argmax(prediction)
    predicted_word = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction)

    return predicted_word, confidence

def test_model(model, label_encoder):
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    recognized_words = []
    frames = []
    predicted_word=""
    confidence=1
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

        if len(frames) == 5:

            predicted_word, confidence = process_frames(frames, model, label_encoder)
            frames = [] 

            if confidence > 0.85:
                recognized_words.append(predicted_word)
                recognized_words = recognized_words[-3:]
                confidence = int(confidence * 100)
        
        cv2.putText(rgb_frame, f"{predicted_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        mp_drawing.draw_landmarks(rgb_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Gesture Recognition", rgb_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # sentence = form_sentence(recognized_words)
    # print("Final Sentence:", sentence)


model = load_model('hand_model.h5')
label_encoder = joblib.load('label_encoder.joblib')

test_model(model, label_encoder)