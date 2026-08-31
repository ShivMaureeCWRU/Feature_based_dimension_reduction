import cv2
import mediapipe as mp
import numpy as np
import time

from joblib import load



model = load("../models/multinomial_logistic_regression_model.joblib")
scaler = load("../models/logistic_regression_scaler.joblib")
label_encoder = load("../models/label_encoder.joblib")



labels_dict = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
    24: "Y", 25: "Z", 26: "Space", 27: "Clear", 28: "Enter"
}


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

current_word = []
current_sentence = ""
current_display = ""

last_prediction_time = time.time()
prediction_interval = 2


while True:
    data_aux = []
    x_ = []
    y_ = []

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
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for i in range(21):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(21):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        if len(data_aux) == 42:
            features = np.asarray(data_aux).reshape(1, -1)

            # so the live features must be scaled before prediction.
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)

            decoded_label = label_encoder.inverse_transform(prediction)[0]

            # If labels were already "A", "B", "C", this fallback handles that too.
            try:
                current_display = labels_dict[int(decoded_label)]
            except ValueError:
                current_display = str(decoded_label)

            print("Prediction:", prediction)
            print("Decoded label:", decoded_label)
            print("Current display:", current_display)
            print("-" * 30)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            cv2.putText(
                frame,
                current_display,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

            if time.time() - last_prediction_time >= prediction_interval:
                if current_display == "Space":
                    current_sentence += "".join(current_word) + " "
                    current_word = []

                elif current_display == "Enter":
                    print("Final Sentence:", current_sentence.strip())
                    current_sentence = ""
                    current_word = []

                elif current_display == "Clear":
                    if current_word:
                        current_word.pop()
                    elif current_sentence:
                        current_sentence = current_sentence.rstrip()
                        current_sentence = current_sentence[:-1]

                    print("Updated Sentence:", current_sentence + "".join(current_word))

                else:
                    current_word.append(current_display)

                last_prediction_time = time.time()

    cv2.putText(
        frame,
        f"Currently Showing: {current_display}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Sentence: {current_sentence + ''.join(current_word)}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Sign Language to Text - Logistic Regression", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()