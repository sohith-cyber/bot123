import numpy as np
import pickle
import json
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

# Load preprocessed data
words = pickle.load(open("C:/Users/velag/Desktop/Music-me-Chatbot_song_recommendor_system--main/files required/words.pkl", 'rb'))
classes = pickle.load(open("C:/Users/velag/Desktop/Music-me-Chatbot_song_recommendor_system--main/files required/classes.pkl", 'rb'))
model = load_model("C:/Users/velag/Desktop/Music-me-Chatbot_song_recommendor_system--main/files required/model.h5")

# Load intents file
intents = json.loads(open("C:/Users/velag/Desktop/Music-me-Chatbot_song_recommendor_system--main/files required/intents.json").read())

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word and remove stopwords
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in stopwords.words('english')]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))

def predict_class(sentence):
    # Filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        # Return a default response if no intent is detected
        return "I'm sorry, I didn't understand your request. Could you please rephrase or provide more context?"

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    else:
        # Return a default response if no matching intent is found
        result = "I'm afraid I don't have a specific response for that. Let me know if you have any other questions."

    return result

from emotion import predict_emotion_list
# Variable to keep track of the conversation state
conversation_started = False
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    global conversation_started
    data = request.json
    user_input = data['user_input']
    user_messages = data.get('userMessages', [])  # Get the user messages from the request

    # If the user wants to quit the conversation, stop it and print the user messages
    if user_input.lower() == "quit":
        conversation_started = False
        print("User messages:", user_messages)
        print(predict_emotion_list(user_messages))
        return jsonify({'response': "Conversation ended. Have a great day!"})

    # If the conversation has not started yet or the user input is empty, initiate it with a greeting message
    if not conversation_started or not user_input:
        conversation_started = True
        response = "Hello, how's your day so far?"
    else:
        # Get chatbot's response
        ints = predict_class(user_input)
        response = get_response(ints, intents)

    return jsonify({'response': response})

@app.route('/get_emotion_probabilities', methods=['POST'])
def get_emotion_probabilities():
    data = request.json
    messages = data['messages']
    emotion_probabilities = predict_emotion_list(messages)
    return jsonify({'emotion_probabilities': emotion_probabilities})

@app.route('/get_voice_response', methods=['POST'])
def get_voice_response():
    global conversation_started
    # Get the voice input from the request
    audio_data = request.data
    
    # Use speech recognition to convert the voice input to text
    r = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = r.record(source)
    user_input = r.recognize_google(audio)
    
    # Call the existing get_bot_response function to handle the text input
    return get_bot_response()
    
    # If the conversation has not started yet or the user input is empty, initiate it with a greeting message
    if not conversation_started or not user_input:
        conversation_started = True
        response = "Hello, how's your day so far?"
    else:
        # Get chatbot's response
        ints = predict_class(user_input)
        response = get_response(ints, intents)
    
    return jsonify({'response': response})

# Route to send an initial request when the page is loaded
@app.route('/initiate_conversation', methods=['GET'])
def initiate_conversation():
    global conversation_started
    conversation_started = False
    response = "Hello, how's your day so far?"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)