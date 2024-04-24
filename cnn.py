import librosa
import librosa.display
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_sound_folder_csv(folder_path, csv_file_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print("Folder not found.")
        return None
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print("CSV file not found.")
        return None
    
    # Get list of sound files in the folder
    sound_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    
    # Create a DataFrame to store information about the sound files
    sound_data_csv = pd.DataFrame(columns=['File Name', 'Category'])
    
    for file_name in sound_files:
        if file_name in df['filename'].values:
            category = df[df['filename'] == file_name]['category'].values[0]
            sound_data_csv = sound_data_csv._append({'File Name': file_name, 'Category': category}, ignore_index=True)
    
    return sound_data_csv

def extract_mfcc(file_path, n_mfcc=40, n_fft=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000)   #sr=16000 n_mfcc = 40 in case
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)  
    mfcc = mfcc.T
    
    return mfcc

def apply_mfcc_to_folder(folder_path):
    sound_data = pd.DataFrame(columns=['File Name', 'MFCC'])
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav') or file_name.endswith('.mp3'):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc(file_path)
            sound_data = sound_data._append({'File Name': file_name, 'MFCC': mfcc}, ignore_index=True)
    
    return sound_data


folder_path = 'D:/HUST/dev/py/sound_classification/audio_train'
csv_file_path = 'D:/HUST/dev/py/sound_classification/bee_sound.csv'
sound_data_csv = read_sound_folder_csv(folder_path, csv_file_path)
mfcc_data = apply_mfcc_to_folder(folder_path)


# Prepare data
X = []
y = []
for idx, row in mfcc_data.iterrows():
    if len(row['MFCC']) == 0:
        print(f"Ignoring {row['File Name']}: No MFCC data available.")
        continue
    X.append(row['MFCC'])
    y.append(sound_data_csv.loc[idx, 'Category'])

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for CNN
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(63,40,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history= model.fit(X_train_cnn, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))
model.save('D:/HUST/dev/py/sound_classification/model/sound_classification.keras')



#metrics
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_cnn, y_test)
print('Test accuracy:', test_acc)

# Predict labels for the test data
predictions = model.predict(X_test_cnn)
predicted_labels = np.argmax(predictions, axis=1)    #check

# Decode labels
predicted_labels = le.inverse_transform(predicted_labels)
true_labels = le.inverse_transform(y_test)

# Print classification report
print(classification_report(true_labels, predicted_labels))

# Plot confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()







