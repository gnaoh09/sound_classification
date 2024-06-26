import librosa
import numpy as np
from keras.models import load_model

class Config(object):
    def __init__(self, sampling_rate=44100, audio_duration=2, n_mfcc=40):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_mfcc = n_mfcc
        self.audio_length = self.sampling_rate * self.audio_duration
        self.dim = ( 1 + int(np.floor(self.audio_length / 512)),self.n_mfcc, 1)

def preprocessing(fname, audio_duration=2, sr=16000, n_mfcc=40, total_features=80):
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
        mfcc = mfcc.reshape((1,mfcc.shape[0], mfcc.shape[1],1))
        X[i] = mfcc.T   
    return X



model = load_model("D:/HUST/dev/py/sound_classification/model/sound_classification.keras")
fname = ["D:/HUST/dev/py/sound_classification/audio_train/queen_49.wav"]
data = preprocessing(fname)
predictions = model.predict(data)
for x in predictions:
    if x < 0.1:
        print("queen")
    else:
        print("no queen")
print(predictions)


