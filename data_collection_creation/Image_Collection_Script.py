import os  # For working with file system paths and directories
import cv2  # For handling image capture and processing

DATA_DIR = '../data_2'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 200

print("Checking for available webcams...")
webcam_index = None

for i in range(6):  # Looping through to find camera index
    cap = cv2.VideoCapture(i)
    if cap.isOpened():  # Check if the webcam at index `i` is functional
        print(f"Webcam {i} is available.")
        cap.release()
        webcam_index = i
        break
    cap.release()

if webcam_index is None:
    print("No webcams found. Exiting...")
    exit()

cap = cv2.VideoCapture(webcam_index)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            print("Failed to capture frame. Webcam Error.")
            break
        cv2.putText(frame, 'Ready Freddy? Press S !',
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (20, 0, 150), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):  # Wait for user to press 's' to start
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        cv2.imshow('frame', frame)  # Display the frame to the user
        cv2.waitKey(25)  # Wait briefly to simulate frame rate

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
