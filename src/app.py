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
import os
from dotenv import load_dotenv
from emotion import predict_emotion_list

app = Flask(__name__, static_url_path='/static')
CORS(app)
load_dotenv()

# PostgreSQL Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://root:At51RyPT7W0xQhm4j8U9jmzBHTg4jkaC@dpg-cpphes88fa8c739galk0-a.oregon-postgres.render.com/bot2'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY')
db = SQLAlchemy(app)

# Define models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class EmotionProbability(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
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

@app.route('/index')
def index():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return render_template('index.html', username=user.username)
    return redirect(url_for('login'))

@app.route('/about')
def about():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return render_template('about.html', username=user.username)
    return redirect(url_for('login'))

@app.route('/therapy')
def therapy():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            today_date = datetime.date.today()
            emotion_probability_record = EmotionProbability.query.filter_by(username=user.username, date=today_date).first()

            if emotion_probability_record:
                emotion_probabilities = json.loads(emotion_probability_record.emotion_probability)
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

            return render_template('therapy.html', username=user.username, playlist_name=playlist_name, spotify_client_id=spotify_client_id, spotify_secret_id=spotify_secret_id)

    return redirect(url_for('login'))

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
            return redirect(url_for('login'))
    return render_template('login.html', form=form)

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return render_template('dashboard.html', user=user)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the required files
words_path = os.path.join( 'templates', 'words.pkl')
classes_path = os.path.join( 'templates', 'classes.pkl')
model_path = os.path.join( 'templates', 'chatbot_model.h5')
intents_path = os.path.join( 'templates', 'intents.json')

# Load preprocessed data
with open(words_path, 'rb') as words_file:
    words = pickle.load(words_file)

with open(classes_path, 'rb') as classes_file:
    classes = pickle.load(classes_file)

# Load model
model = load_model(model_path)

# Load intents file
with open(intents_path, 'r') as intents_file:
    intents = json.loads(intents_file.read())

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

    return np.array(bag)

def predict_class(sentence):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
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

# Variable to keep track of the conversation state
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
        if user:
            emotion_data = EmotionProbability.query.filter_by(username=user.username).order_by(EmotionProbability.date.desc()).all()
            user_messages = ['Hello', 'How are you?', 'I am fine', 'Thank you']  # Replace this with actual user messages

            formatted_emotion_data = []
            for record in emotion_data:
                formatted_emotion_probability = record.emotion_probability.replace("{", "").replace("}", "").replace("'", "").replace('"', '')
                formatted_emotion_data.append((record.id, record.date, formatted_emotion_probability))

            return render_template('monitor_emotions.html', emotion_data=formatted_emotion_data, userMessages=user_messages, username=user.username)
    
    flash("Please log in to monitor your emotions.")
    return redirect(url_for('login'))

@app.route('/mood_tracker', methods=['GET'])
def mood_tracker():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
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

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
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

@app.route('/get_emotion_data', methods=['GET'])
def get_emotion_data():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            emotion_data = EmotionProbability.query.filter_by(username=user.username).order_by(EmotionProbability.date.desc()).limit(5).all()
            
            formatted_emotion_data = []
            for record in emotion_data:
                emotion_probability = json.loads(record.emotion_probability)
                positive = sum([value for key, value in emotion_probability.items() if key in ['joy', 'surprise']])
                negative = sum([value for key, value in emotion_probability.items() if key not in ['joy', 'surprise']])
                total = positive + negative
                
                if total > 0:
                    positive1 = (positive / total) * 100
                    negative1 = (negative / total) * 100
                    formatted_emotion_data.append({
                        'date': record.date.strftime('%Y-%m-%d'),
                        'positive': positive1,
                        'negative': negative1
                    })
            
            formatted_emotion_data.reverse()  # To show the oldest date first
            return jsonify(formatted_emotion_data)
    
    return jsonify([]), 401

if __name__ == '__main__':
    app.run(debug=True)
