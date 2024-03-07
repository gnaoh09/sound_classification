from sklearn.metrics import accuracy_score
import pickle
from sound_c import X_test, y_test


with open('D:/HUST/dev/py/bee_classification/models/MFCCs_80_RF_60k.pkl', 'rb') as f:
    pickle_model = pickle.load(f)
    # Make predictions using the loaded model
y_pred_pickle = pickle_model.predict(X_test)

# Calculate accuracy for the pickle model
accuracy_pickle = accuracy_score(y_test, y_pred_pickle)
print("Accuracy for pickle model:", accuracy_pickle)