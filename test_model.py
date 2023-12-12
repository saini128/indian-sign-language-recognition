import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import joblib  # For scikit-learn versions < 0.21
# For scikit-learn versions >= 0.21, you can use: from joblib import dump, load
def test_model(model,label_encoder):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    # Set up the Holistic model with appropriate parameters
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp_hands.Hands()
    pose = mp_pose.Pose()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # Check if at least one hand is detected with sufficient confidence

        input_features = np.concatenate([pose, lh, rh])


        # Make predictions using the trained model
        prediction = model.predict(np.array([input_features]))
        predicted_class = np.argmax(prediction)
        predicted_word = label_encoder.inverse_transform([predicted_class])[0]

        # Display the predicted word
        cv2.putText(rgb_frame, predicted_word, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        mp_drawing.draw_landmarks(rgb_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # Display the frame
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Gesture Recognition", rgb_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load the trained model
model = load_model('hand_model.h5')

# Load the LabelEncoder
label_encoder = joblib.load('label_encoder.joblib')

# Use your testing function with the loaded model and LabelEncoder
test_model(model, label_encoder)