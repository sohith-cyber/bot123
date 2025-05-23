import os
from flask import Flask, render_template, redirect, url_for, session, flash, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import pickle
import json
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import random
import speech_recognition as sr
import datetime
from dotenv import load_dotenv
from emotion import predict_emotion_list
from flask import jsonify
import json

# Load environment variables
load_dotenv()

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Velaga%4096@127.0.0.1:3306/bot'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')

db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Define EmotionProbability model
class EmotionProbability(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), db.ForeignKey('user.username'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    emotion_probability = db.Column(db.String(255), nullable=False)

# Create tables
with app.app_context():
    db.create_all()

class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        user = User.query.filter_by(email=field.data).first()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.hashpw(form.password.data.encode('utf-8'), bcrypt.gensalt())
        new_user = User(username=form.name.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.checkpw(form.password.data.encode('utf-8'), user.password.encode('utf-8')):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash("Login failed. Please check your email and password")
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return render_template('index.html', username=user.username)
    return redirect(url_for('login'))

@app.route('/about')
def about():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return render_template('about.html', username=user.username)
    return redirect(url_for('login'))

@app.route('/therapy')
def therapy():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        today_date = datetime.date.today()
        emotion_prob = EmotionProbability.query.filter_by(username=user.username, date=today_date).first()
        
        if emotion_prob:
            emotion_probabilities = json.loads(emotion_prob.emotion_probability)
        else:
            emotion_probabilities = {}

        highest_emotion = max(emotion_probabilities, key=emotion_probabilities.get) if emotion_probabilities else None

        playlist_map = {
            'joy': 'Happy Playlist',
            'anger': 'Angry Playlist',
            'fear': 'Calming Playlist',
            'sadness': 'Motivational Playlist',
            'surprise': 'Upbeat Playlist',
            'disgust': 'Relaxing Playlist'
        }

        playlist_name = playlist_map.get(highest_emotion, 'General Playlist')
        spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
        spotify_secret_id = os.getenv('SPOTIFY_CLIENT_SECRET')

        return render_template('therapy.html', username=user.username, playlist_name=playlist_name, 
                               spotify_client_id=spotify_client_id, spotify_secret_id=spotify_secret_id)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return render_template('dashboard.html', user=user)
    return redirect(url_for('login'))

# Load chatbot model and data
base_dir = os.path.dirname(os.path.abspath(__file__))
words_path = os.path.join(base_dir, 'templates', 'words.pkl')
classes_path = os.path.join(base_dir, 'templates', 'classes.pkl')
model_path = os.path.join(base_dir, 'templates', 'chatbot_model.h5')
intents_path = os.path.join(base_dir, 'templates', 'intents.json')

with open(words_path, 'rb') as words_file:
    words = pickle.load(words_file)

with open(classes_path, 'rb') as classes_file:
    classes = pickle.load(classes_file)

model = load_model(model_path)

with open(intents_path, 'r') as intents_file:
    intents = json.loads(intents_file.read())

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in stopwords.words('english')]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't understand your request. Could you please rephrase or provide more context?"
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm afraid I don't have a specific response for that. Let me know if you have any other questions."
    return result

conversation_started = False

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    global conversation_started
    data = request.json
    user_input = data['user_input']
    user_messages = data.get('userMessages', [])

    if user_input.lower() == "quit":
        conversation_started = False
        print("User messages:", user_messages)
        print(predict_emotion_list(user_messages))
        return jsonify({'response': "Conversation ended. Have a great day!"})

    if not conversation_started or not user_input:
        conversation_started = True
        response = "Hello, how's your day so far?"
    else:
        ints = predict_class(user_input)
        response = get_response(ints, intents)

    return jsonify({'response': response})

@app.route('/monitor_emotions', methods=['GET'])
def monitor_emotions():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        emotion_data = EmotionProbability.query.filter_by(username=user.username).all()

        user_messages = ['Hello', 'How are you?', 'I am fine', 'Thank you']  # Replace this with actual user messages

        formatted_emotion_data = []
        for row in emotion_data:
            formatted_emotion_data.append((row.id, row.date.strftime('%Y-%m-%d'), row.emotion_probability))

        return render_template('monitor_emotions.html', emotion_data=formatted_emotion_data, userMessages=user_messages, username=user.username)
    
    flash("Please log in to monitor your emotions.")
    return redirect(url_for('login'))

@app.route('/mood_tracker', methods=['GET'])
def mood_tracker():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        messages = request.args.get('messages')
        if messages:
            messages = json.loads(messages)
            emotion_probabilities = predict_emotion_list(messages)
        else:
            emotion_probabilities = {}

        today_date = datetime.date.today()
        existing_record = EmotionProbability.query.filter_by(username=user.username, date=today_date).first()

        if existing_record:
            existing_probabilities = json.loads(existing_record.emotion_probability)
            updated_probabilities = {}

            for emotion in emotion_probabilities:
                if emotion in existing_probabilities:
                    updated_probabilities[emotion] = (emotion_probabilities[emotion] + existing_probabilities[emotion]) / 2
                else:
                    updated_probabilities[emotion] = emotion_probabilities[emotion]

            for emotion in existing_probabilities:
                if emotion not in updated_probabilities:
                    updated_probabilities[emotion] = existing_probabilities[emotion]

            existing_record.emotion_probability = json.dumps(updated_probabilities)
        else:
            new_record = EmotionProbability(username=user.username, date=today_date, emotion_probability=json.dumps(emotion_probabilities))
            db.session.add(new_record)

        db.session.commit()

        return render_template('mood_tracker.html', emotion_probabilities=emotion_probabilities, username=user.username)

    flash("Please log in to use the mood tracker.")
    return redirect(url_for('login'))

@app.route('/get_emotion_probabilities', methods=['GET'])
def get_emotion_probabilities_get():
    messages = request.args.get('messages')
    if messages:
        messages = json.loads(messages)
        emotion_probabilities = predict_emotion_list(messages)
        formatted_emotion_probabilities = {emotion: round(probability, 2) for emotion, probability in emotion_probabilities.items()}
        return jsonify({'emotion_probabilities': formatted_emotion_probabilities})
    else:
        return jsonify({'emotion_probabilities': {}})

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        
        if user:
            EmotionProbability.query.filter_by(username=user.username).delete()
            db.session.delete(user)
            db.session.commit()

            session.pop('user_id', None)
            flash("Your account has been deleted successfully.")

            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'User not found'}), 400

    return jsonify({'status': 'error', 'message': 'User not logged in'}), 400

@app.route('/get_voice_response', methods=['POST'])
def get_voice_response():
    global conversation_started
    audio_data = request.data
    
    r = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = r.record(source)
    user_input = r.recognize_google(audio)
    
    return get_bot_response()

@app.route('/get_emotion_data', methods=['GET'])
def get_emotion_data():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        
        emotion_data = EmotionProbability.query.filter_by(username=user.username)\
            .order_by(EmotionProbability.date.desc())\
            .limit(5)\
            .all()
        
        formatted_emotion_data = []
        for record in emotion_data:
            emotion_probability = json.loads(record.emotion_probability)
            positive = sum([value for key, value in emotion_probability.items() if key in ['joy', 'surprise']])
            negative = sum([value for key, value in emotion_probability.items() if key not in ['joy', 'surprise']])
            total = positive + negative
            
            if total > 0:
                positive_percentage = (positive / total) * 100
                negative_percentage = (negative / total) * 100
                formatted_emotion_data.append({
                    'date': record.date.strftime('%Y-%m-%d'),
                    'positive': positive_percentage,
                    'negative': negative_percentage
                })
        
        formatted_emotion_data.reverse()  # To show the oldest date first
        return jsonify(formatted_emotion_data)
    
    return jsonify([]), 401
@app.route('/initiate_conversation', methods=['GET'])
def initiate_conversation():
    global conversation_started
    conversation_started = True
    response = "Hello, how's your day so far?"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
