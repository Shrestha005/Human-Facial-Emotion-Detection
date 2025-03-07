import os

dataset_path = "dataset/train"
emotions = os.listdir(dataset_path)

for emotion in emotions:
    images = os.listdir(os.path.join(dataset_path, emotion))
    print(f"{emotion}: {len(images)} images")
