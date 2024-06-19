import numpy as np
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import os


# Load the saved model
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
model_path = os.path.join(base_dir, 'templates', 'Emotion Recognition.h5')

# Load the model
model = load_model(model_path)
# Load the tokenizer
# Define the path to tokenizer.pickle using os.path.join for platform independence
tokenizer_path = os.path.join(base_dir, 'templates', 'tokenizer.pickle')

# Load the tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Define the list of emotion names
emotion_names = ['anger', 'disgust', 'joy', 'fear', 'sadness', 'surprise']

# Function to clean text
def clean(text):
    # Your text cleaning code here
    return text

# Function to predict emotion
def predict_emotion_list(messages):
    emotion_probabilities_sum = {emotion: 0 for emotion in emotion_names}
    emotion_counts = {emotion: 0 for emotion in emotion_names}

    for text in messages[:-1]:  # Exclude the last index message
        cleaned_text = clean(text)
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequences = np.array(pad_sequences(sequences, maxlen=256, truncating='pre'))
        
        predictions = model.predict(padded_sequences)[0]
        emotion_probabilities = dict(zip(emotion_names, predictions))
        
        for emotion, probability in emotion_probabilities.items():
            emotion_probabilities_sum[emotion] += probability
            emotion_counts[emotion] += 1

    emotion_probabilities_avg = {emotion: emotion_probabilities_sum[emotion] / max(1, emotion_counts[emotion]) for emotion in emotion_names}
    return emotion_probabilities_avg





