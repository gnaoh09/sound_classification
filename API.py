import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from pydantic import BaseModel
import os
import uvicorn  
from tempfile import NamedTemporaryFile
app = FastAPI()

MODEL_PATH = 'D:/HUST/dev/py/sound_classification/model/sound_classification.keras'


class Config(object):
    def __init__(self, sampling_rate= 16000, audio_duration=20, n_mfcc=40):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_mfcc = n_mfcc
        self.audio_length = self.sampling_rate * self.audio_duration
        self.dim = ( 1 + int(np.floor(self.audio_length / 512)),self.n_mfcc, 1)

def preprocessing(file_path, audio_duration=20, sr=16000, n_mfcc=40, n_fft=2048):
    config = Config(sampling_rate=sr, audio_duration=audio_duration, n_mfcc=n_mfcc)
    input_length = config.audio_length
    X = np.empty(shape=(len(file_path),) + (config.dim)) 
    data, _ = librosa.core.load(file_path, sr=config.sampling_rate)
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

    mfcc = librosa.feature.mfcc(y=data, sr=config.sampling_rate, n_mfcc=config.n_mfcc, n_fft=n_fft)
    mfcc = mfcc.reshape((1,mfcc.shape[0], mfcc.shape[1],1))
    X = mfcc.T  

    return X

@app.post("/predict/")
async def predict_category(file:UploadFile):
    try:
        
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_name = tmp.name
        
        
        data = preprocessing(tmp_name)                        
        model = load_model(MODEL_PATH)
        # Make prediction   
        predictions = model.predict(data)
        print(predictions)
        for x in predictions:
            if x > 0.9:
                category = "no queen"
            elif x < 0.03:
                category = "queen"
            else:
                category = "unknown"
        return {"category": category},200
    
    except Exception as e:
        return {"error": str(e)},500

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0",
                port=8001, log_level="debug")