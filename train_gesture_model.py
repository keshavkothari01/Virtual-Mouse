import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load image data and labels from JSON files


def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        labels_dict = json.load(f)

    images = []
    labels = []

    for img_path, label in labels_dict.items():
        if os.path.isfile(img_path):  # Use img_path directly without appending folder path
            image = load_img(img_path, target_size=(64, 64))
            image = img_to_array(image) / 255.0
            images.append(image)
            labels.append(label)

    print(f"Loaded {len(images)} images for label '{label}' from {json_file}")
    return np.array(images), np.array(labels)


# Load data for each gesture
gesture_labels = ['left_click', 'right_click',
                  'scroll_up', 'scroll_down', 'screenshot']
X, y = [], []

for gesture in gesture_labels:
    json_file = f'gesture_frames/{gesture}_labels.json'
    if os.path.isfile(json_file):
        images, labels = load_data_from_json(json_file)
        if images.size > 0:
            X.append(images)
            y += [gesture] * len(images)
        else:
            print(f"Warning: No images loaded for gesture '{
                  gesture}'. Check {json_file}")
    else:
        print(f"Warning: JSON file for '{gesture}' not found at {json_file}")

# Verify if data is loaded
if len(X) == 0 or len(y) == 0:
    raise ValueError(
        "No data loaded. Please check if the JSON files and images are in the correct location.")

X = np.concatenate(X, axis=0)
y = np.array(y)

print(f"Total images loaded: {X.shape[0]}")
print(f"Total labels loaded: {y.shape[0]}")

# Encode labels
label_mapping = {gesture: idx for idx, gesture in enumerate(gesture_labels)}
y_encoded = np.array([label_mapping[label] for label in y])
y_categorical = to_categorical(y_encoded, num_classes=len(gesture_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(gesture_labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

# Save the model
model.save('gesture_recognition_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {accuracy:.2f}")
