import cv2
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
from mediapipe import solutions as mp_solutions

# Load the trained CNN model
model = load_model('gesture_recognition_model.h5')

# Initialize MediaPipe hands
mp_hands = mp_solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp_solutions.drawing_utils

# Define gesture mappings
gesture_mapping = {0: 'left_click', 1: 'right_click',
                   2: 'scroll_up', 3: 'scroll_down', 4: 'screenshot'}

# Define screen resolution
screen_width, screen_height = pyautogui.size()


def get_gesture_from_model(frame):
    # Pre-process the frame for CNN model
    frame_resized = cv2.resize(frame, (64, 64))
    frame_array = np.expand_dims(frame_resized, axis=0) / 255.0
    predictions = model.predict(frame_array)
    gesture_class = np.argmax(predictions)
    return gesture_mapping.get(gesture_class, 'unknown')


def process_frame(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the 10th landmark for mouse movement (base of the middle finger)
            landmark = landmarks.landmark[9]  # Adjust index if needed
            x, y = int(
                landmark.x * screen_width), int(landmark.y * screen_height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Get the gesture
            gesture = get_gesture_from_model(frame)

            # Perform actions based on the gesture
            if gesture == 'left_click':
                pyautogui.click(x, y)
            elif gesture == 'right_click':
                pyautogui.rightClick(x, y)
            elif gesture == 'scroll_up':
                pyautogui.scroll(100)
            elif gesture == 'scroll_down':
                pyautogui.scroll(-100)
            elif gesture == 'screenshot':
                pyautogui.screenshot('screenshot.png')
                print("Screenshot saved as screenshot.png")

    return frame


def main():
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame = process_frame(frame)

        # Display the frame
        cv2.imshow('Virtual Mouse Control', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
