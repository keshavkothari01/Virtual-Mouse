import cv2
import time
import os

# Define gesture names
gestures = ['left_click', 'right_click',
            'scroll_up', 'scroll_down', 'screenshot']
save_directory = 'gestures_videos'  # Folder where videos will be saved


# Create the folder if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define recording parameters
width, height = 1280, 720
fps = 20.0
record_duration = 60  # Duration in seconds (1 minute)

# Function to record video for a given gesture


def record_gesture_video(gesture_name):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f'{save_directory}/{gesture_name}.mp4', fourcc, fps, (width, height))

    print(f"Recording {gesture_name} gesture for {record_duration} seconds.")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Recording - {gesture_name}', frame)
            out.write(frame)

            # Check if recording time has exceeded the specified duration
            if time.time() - start_time >= record_duration:
                break

            # Press 'q' to stop early if needed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"{gesture_name} gesture recorded and saved to {
          save_directory}/{gesture_name}.mp4")


# Loop through each gesture
for gesture in gestures:
    input(f"Press Enter to start recording for '{gesture}' gesture...")
    record_gesture_video(gesture)
    print(f"Recording for '{gesture}' completed.\n")
