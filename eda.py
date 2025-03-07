import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset path
dataset_path = "dataset/train"

# Emotion labels
emotions = os.listdir(dataset_path)
emotion_counts = {}

# Count images per category
for emotion in emotions:
    emotion_counts[emotion] = len(os.listdir(os.path.join(dataset_path, emotion)))

# Plot the distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=list(emotion_counts.keys()), y=list(emotion_counts.values()))
plt.xlabel("Emotions")
plt.ylabel("Number of Images")
plt.title("Dataset Distribution")
plt.xticks(rotation=45)
plt.show()
