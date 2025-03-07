# import cv2
# import os
# import numpy as np
# import pickle

# # Define dataset paths
# train_path = "dataset/train"
# test_path = "dataset/test"

# # Image size (FER-2013 images are 48x48)
# IMG_SIZE = 48

# # Function to preprocess images
# def preprocess_images(data_path):
#     processed_data = []
#     labels = []
#     emotions = os.listdir(data_path)  # Emotion categories
    
#     for label, emotion in enumerate(emotions):
#         emotion_path = os.path.join(data_path, emotion)
        
#         for img_name in os.listdir(emotion_path):
#             img_path = os.path.join(emotion_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 48x48
            
#             processed_data.append(img)
#             labels.append(label)
    
#     return np.array(processed_data), np.array(labels)

# # Preprocess train and test datasets
# X_train, y_train = preprocess_images(train_path)
# X_test, y_test = preprocess_images(test_path)

# # Normalize pixel values (scale between 0 and 1)
# X_train, X_test = X_train / 255.0, X_test / 255.0

# # Save the preprocessed data using pickle
# with open("X_train.pkl", "wb") as f:
#     pickle.dump(X_train, f)
# with open("y_train.pkl", "wb") as f:
#     pickle.dump(y_train, f)
# with open("X_test.pkl", "wb") as f:
#     pickle.dump(X_test, f)
# with open("y_test.pkl", "wb") as f:
#     pickle.dump(y_test, f)

# print("Preprocessing complete. Data saved!")


import cv2
import os
import numpy as np
import pickle

# Define dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

# Image size (FER-2013 images are 48x48)
IMG_SIZE = 48

# Function to preprocess images
def preprocess_images(data_path):
    processed_data = []
    labels = []
    
    # Check if dataset path exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset path '{data_path}' not found!")
        return None, None

    emotions = os.listdir(data_path)  # Emotion categories
    
    for label, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_path, emotion)

        if not os.path.isdir(emotion_path):
            print(f"Skipping: {emotion_path} (not a directory)")
            continue

        print(f"Processing {emotion}...")  # Track progress

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            
            # Check if the file is an image
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping: {img_name} (not an image)")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping...")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 48x48
            processed_data.append(img)
            labels.append(label)

    return np.array(processed_data), np.array(labels)

# Preprocess train and test datasets
X_train, y_train = preprocess_images(train_path)
X_test, y_test = preprocess_images(test_path)

# If any dataset is empty, stop execution
if X_train is None or X_test is None:
    print("Error: Preprocessing failed. Check dataset path and structure.")
    exit()

# Normalize pixel values (scale between 0 and 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Save the preprocessed data using pickle
with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("✅ Preprocessing complete. Data saved!")
