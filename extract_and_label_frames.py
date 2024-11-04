import cv2
import os
import json
import numpy as np
import mediapipe as mp

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1, min_detection_confidence=0.7)


# Use higher resolution for better clarity
def extract_frames(video_file, output_folder, label, image_size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    labels = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Keep original size for better visibility, resize only for model input
                blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
                mp_drawing.draw_landmarks(blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(0, 255, 0), thickness=4, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3))
                frame_file = os.path.join(output_folder, f"{label}_frame_{
                                          frame_count:04d}.jpg")
                cv2.imwrite(frame_file, blank_image)
                labels[frame_file] = label

        frame_count += 1

    cap.release()
    with open(os.path.join(output_folder, f'{label}_labels.json'), 'w') as f:
        json.dump(labels, f, indent=4)


# Extract frames for each gesture video
gesture_videos = {
    'navigation': 'gestures_videos/navigation.mp4',
    'left_click': 'gestures_videos/left_click.mp4',
    'right_click': 'gestures_videos/right_click.mp4',
    'scroll_up': 'gestures_videos/scroll_up.mp4',
    'scroll_down': 'gestures_videos/scroll_down.mp4',
    'screenshot': 'gestures_videos/screenshot.mp4'
}

output_folder = 'gesture_frames'
for label, video_file in gesture_videos.items():
    extract_frames(video_file, output_folder, label)
