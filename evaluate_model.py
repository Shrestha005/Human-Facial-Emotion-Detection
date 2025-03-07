import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed test data
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Reshape test data for prediction
X_test = X_test.reshape(-1, 48, 48, 1)  # Ensure correct input shape

# Get predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
