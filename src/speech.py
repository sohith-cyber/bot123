from tensorflow.keras.models import model_from_json
import numpy as np
import pickle
import librosa

# Load the model architecture
json_file = open("C:/Users/velag/Downloads/results/CNN_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("C:/Users/velag/Downloads/results/best_model1_weights.h5")
print("Loaded model from disk")

# Load the scaler
with open('C:/Users/velag/Downloads/results/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

# Load the encoder (only needed for reference)
with open('C:/Users/velag/Downloads/results/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Done")

# Define feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result, zcr(data, frame_length, hop_length), rmse(data, frame_length, hop_length), mfcc(data, sr, frame_length, hop_length)))
    return result

# Function to preprocess a new audio array and make it ready for prediction
def get_predict_feat(audio_array, sr):
    res = extract_features(audio_array, sr=sr)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 2376))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

# Define the emotion labels (adjust based on your dataset)
emotions1 = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fear', 6: 'Disgust'}
print("Emotion labels:", list(emotions1.values()))

# Function to make a prediction and return the probabilities
def prediction(audio_array, sr):
    res = get_predict_feat(audio_array, sr)
    predictions = loaded_model.predict(res)
    print(f"Predictions shape: {predictions.shape}")
    print(predictions)
    return predictions[0]

# Function to average probabilities across multiple audio arrays
def average_probabilities(audio_arrays, sample_rates):
    emotion_probs = {label: 0.0 for label in emotions1.values()}
    num_arrays = len(audio_arrays)

    for audio_array, sr in zip(audio_arrays, sample_rates):
        probs = prediction(audio_array, sr)
        print(probs)
        for i, emotion in emotions1.items():
            emotion_probs[emotion] += probs[i] / num_arrays

    return emotion_probs