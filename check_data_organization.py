import os
import json

# Define expected gesture labels and their JSON file paths
gesture_labels = ['left_click', 'right_click',
                  'scroll_up', 'scroll_down', 'screenshot']
json_folder = 'gesture_frames'

# Check each gesture's JSON file and the associated images
for gesture in gesture_labels:
    json_file_path = os.path.join(json_folder, f"{gesture}_labels.json")

    # Check if JSON file exists
    if not os.path.isfile(json_file_path):
        print(f"Warning: JSON file for '{
              gesture}' not found at {json_file_path}")
        continue

    # Load JSON file content
    with open(json_file_path, 'r') as f:
        labels_dict = json.load(f)

    # Verify each image path in JSON file
    missing_images = []
    for img_path in labels_dict.keys():
        full_img_path = os.path.join(json_folder, img_path)
        if not os.path.isfile(full_img_path):
            missing_images.append(full_img_path)

    # Report results
    if missing_images:
        print(f"Warning: Missing images for gesture '{
              gesture}' in {json_file_path}:")
        for missing_img in missing_images:
            print(f" - {missing_img}")
    else:
        print(f"All images for gesture '{gesture}' are accounted for.")

print("\nCheck completed.")
