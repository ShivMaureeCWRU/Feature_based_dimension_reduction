import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('../models/model.p', 'rb'))  # Load the model dictionary
model = model_dict['model']  # Extract the trained model

cap = cv2.VideoCapture(0)  # Capture video from the first camera (index 0)

mp_hands = mp.solutions.hands  # MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils  # Utilities for drawing
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24:'Y', 25:'Z',26: 'Space', 27:'Clear', 28:'Enter'
}

while True:
    data_aux = []  # Auxiliary list to store normalized landmarks data
    x_ = []  # List to store x-coordinates of landmarks
    y_ = []  # List to store y-coordinates of landmarks

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,  # The image/frame to draw on
            hand_landmarks,  # Detected hand landmarks
            mp_hands.HAND_CONNECTIONS,  # Connections between landmarks
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for i in range(21):  # Limit to 21 landmarks
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)  # Append x-coordinates
            y_.append(y)  # Append y-coordinates

        for i in range(21):  # Limit to 21 landmarks
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])  # Make prediction based on landmarks data
            predicted_character = labels_dict[int(prediction[0])]  # Map the prediction to the corresponding label

            x1 = int(min(x_) * W) - 10  # Top-left x-coordinate of the bounding box
            y1 = int(min(y_) * H) - 10  # Top-left y-coordinate of the bounding box
            x2 = int(max(x_) * W) - 10  # Bottom-right x-coordinate of the bounding box
            y2 = int(max(y_) * H) - 10  # Bottom-right y-coordinate of the bounding box

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw bounding box
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 3), 3,
                        cv2.LINE_AA)  # Display predicted character above the bounding box
        else:
            print(f"Error: Feature vector size mismatch. Expected 42, got {len(data_aux)}.")

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
