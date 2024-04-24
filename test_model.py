import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
from keras.models import load_model
# Load the model
model = load_model("D:/HUST/dev/py/sound_classification/model/sound_classification_lstm.keras")
class Config(object):
    def __init__(self, sampling_rate=44100, audio_duration=2, n_mfcc=40):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_mfcc = n_mfcc
        self.audio_length = self.sampling_rate * self.audio_duration
        self.dim = ( 1,1 + int(np.floor(self.audio_length / 512)),self.n_mfcc,1)
# Define a function to preprocess a single audio file
def preprocessing(fname, audio_duration=2, sr=16000, n_mfcc=40):
    config = Config(sampling_rate=sr, audio_duration=audio_duration, n_mfcc=n_mfcc)
    X = np.empty(shape=(len(fname),) + (config.dim))  
    input_length = config.audio_length
    for i, file_name in enumerate(fname):
        data, _ = librosa.core.load(file_name, sr=config.sampling_rate, res_type="kaiser_fast")
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        mfcc = librosa.feature.mfcc(y=data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        mfcc = mfcc.reshape((1, mfcc.shape[0], mfcc.shape[1],1))  # Reshape to match the model input shape
        X[i] = mfcc.T
    return X

def preprocess_folder(folder_path):
    X = []
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfcc = preprocessing(file_path)
            X.append(mfcc)
            file_names.append(file_name)
    X = np.vstack(X)
    return X, file_names

test_folder_path = "D:/HUST/dev/py/sound_classification/audio_train"

# Preprocess the test data
X_test, test_file_names = preprocess_folder(test_folder_path)

# Predict labels for the test data
predictions = model.predict(X_test)

# Convert predictions to class labels
class_labels = ["queen", "no queen"]
predicted_labels = [class_labels[np.argmax(pred)] for pred in predictions]

# Load ground truth labels for the test data from CSV file
test_labels_df = pd.read_csv("bee_sound.csv")  
test_labels = test_labels_df["category"].tolist()

# Compute evaluation metrics
accuracy = np.mean(np.array(predicted_labels) == np.array(test_labels))
report = classification_report(test_labels, predicted_labels, target_names=class_labels)
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
