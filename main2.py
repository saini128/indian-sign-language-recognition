import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
import joblib
import spacy
import tkinter as tk
from tkinter.scrolledtext import ScrolledText


nlp = spacy.load("en_core_web_sm")


root = tk.Tk()

def form_sentence(recognized_words):
    
    if not recognized_words:
        return ""

    
    cleaned_words = [recognized_words[0]] + [word for i, word in enumerate(recognized_words[1:], 1) if word != recognized_words[i-1]]

    
    sentence = ' '.join(cleaned_words)
    
    
    doc = nlp(sentence)

    
    formed_sentence = ' '.join([token.text_with_ws for token in doc])
    
    return formed_sentence

def draw_scrollable_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, bg_color=(0, 0, 255), text_color=(255, 255, 255), max_width=400):
    
    bg_img = np.zeros_like(image, dtype=np.uint8)
    bg_img[:40] = bg_color

    
    text_widget = ScrolledText(root, width=50, height=10)
    text_widget.insert(tk.END, text)
    text_widget.xview_moveto(1)
    text = text_widget.get("1.0", tk.END)
    cv2.putText(bg_img, text, tuple(position), font, font_scale, text_color, font_thickness)  

    
    text_width, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width = int(text_width[0])
    
    position = (position[0] - 5, position[1])  
    
    
    if (position[0] + text_width) < 0:
        position = (max_width, position[1])

    
    result = cv2.addWeighted(image, 1, bg_img, 1, 0)

    return result

def test_model(model, label_encoder, confidence_threshold=0.8):
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    recognized_words = []
    max_width = 400

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

        prediction = model.predict(np.array([input_features]))
        predicted_class = np.argmax(prediction)
        predicted_word = label_encoder.inverse_transform([predicted_class])[0]
        confidence = prediction[0][predicted_class]

        if confidence > confidence_threshold:
            recognized_words.append(predicted_word)

        sentence = form_sentence(recognized_words)
        rgb_frame = draw_scrollable_text(rgb_frame, sentence, (10, 30), max_width=max_width)

        mp_drawing.draw_landmarks(rgb_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Gesture Recognition", rgb_frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
    sentence = form_sentence(recognized_words)
    print("Final Sentence:", sentence)

    
model = load_model('hand_model.h5')

    
label_encoder = joblib.load('label_encoder.joblib')

    
test_model(model, label_encoder)


root.mainloop()
