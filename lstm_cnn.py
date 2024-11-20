import librosa
import librosa.display
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

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
    y, sr = librosa.load(file_path, sr=16000)   #sr=16000 n_mfcc = 40 in case
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    return mfcc




def apply_mfcc_to_folder(folder_path):
    mfcc_data = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:  # Check if MFCC extraction was successful
                mfcc_data.append({'File Name': file_name, 'MFCC': mfcc})
            else:
                print(f"Failed to extract MFCC for file: {file_name}")
    
    return mfcc_data




folder_path = 'D:/HUST/dev/py/sound_classification/data/train'
csv_file_path = 'D:/HUST/dev/py/sound_classification/bee_sound_800.csv'
sound_data_csv = read_sound_folder_csv(folder_path, csv_file_path)
mfcc_data = apply_mfcc_to_folder(folder_path)


# Prepare data
X = np.array([row['MFCC'] for row in mfcc_data if len(row['MFCC']) > 0])
y = np.array([sound_data_csv.loc[idx, 'Category'] for idx, row in enumerate(mfcc_data) if len(row['MFCC']) > 0])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN input shape
time_steps, features = X_train.shape[1:]

X_train_cnn_lstm = X_train.reshape((X_train.shape[0], 1, time_steps, features, 1))
X_test_cnn_lstm = X_test.reshape((X_test.shape[0], 1, time_steps, features, 1))


# CNN-LSTM model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None,time_steps, features, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM layers
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_cnn_lstm, y_train, epochs=80, batch_size=32, validation_data=(X_test_cnn_lstm, y_test))
test_loss, test_acc = model.evaluate(X_test_cnn_lstm, y_test)
print('Test accuracy:', test_acc)
# Save the model
model.save('D:/HUST/dev/py/sound_classification/model/sound_classification_lstm.keras')
print('saved')

# Plot training history (optional)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


