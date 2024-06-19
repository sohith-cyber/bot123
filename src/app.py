from flask import Flask, render_template, redirect, url_for, session, flash, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from flask_mysqldb import MySQL
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

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = 'bot'
app.secret_key = os.getenv('SECRET_KEY')
mysql = MySQL(app)

class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")



@app.route('/index')
def index():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            username = user[0]
            return render_template('index.html', username=username)
            
    return redirect(url_for('login'))


@app.route('/about')
def about():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            username = user[0]
            return render_template('about.html', username=username)
            
    return redirect(url_for('login'))
@app.route('/therapy')
def therapy():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            username = user[0]
            
            # Get today's date
            today_date = datetime.date.today()

            # Fetch emotion probabilities for today
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT emotion_probability FROM emotion_probabilities WHERE username=%s AND date=%s", (username, today_date))
            emotion_probability_record = cursor.fetchone()
            cursor.close()

            if emotion_probability_record:
                emotion_probabilities = json.loads(emotion_probability_record[0])
            else:
                emotion_probabilities = {}

            # Determine the highest emotion probability
            highest_emotion = max(emotion_probabilities, key=emotion_probabilities.get) if emotion_probabilities else None

            # Mapping of emotions to playlist names
            playlist_map = {
                'joy': 'Happy Playlist',
                'anger': 'Angry Playlist',
                'fear': 'Calming Playlist',
                'sadness': 'Motivational Playlist',
                'surprise': 'Upbeat Playlist',
                'disgust': 'Relaxing Playlist'
            }

            # Get the corresponding playlist name
            playlist_name = playlist_map.get(highest_emotion, 'General Playlist')
            spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
            spotify_secret_id = os.getenv('SPOTIFY_CLIENT_SECRET')

            return render_template('therapy.html', username=username, playlist_name=playlist_name, spotify_client_id=spotify_client_id,spotify_secret_id=spotify_secret_id)

    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Store data into database 
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_id'] = user[0]
            return redirect(url_for('index'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

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
words_path = os.path.join(base_dir,'src', 'templates', 'words.pkl')
classes_path = os.path.join(base_dir,'src', 'templates', 'classes.pkl')
model_path = os.path.join(base_dir,'src', 'templates', 'chatbot_model.h5')
intents_path = os.path.join(base_dir,'src', 'templates', 'intents.json')

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
    # Check if 'user_id' is in session
    if 'user_id' in session:
        user_id = session['user_id']

        # Fetch the username associated with the user_id
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            username = user[0]

            # Fetch emotion data for the logged-in user
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT id, date, emotion_probability FROM emotion_probabilities WHERE username=%s", (username,))
            emotion_data = cursor.fetchall()
            cursor.close()

            # Fetch the user messages from the database or any other source
            user_messages = ['Hello', 'How are you?', 'I am fine', 'Thank you']  # Replace this with the actual user messages

            # Format the emotion_data to remove time, curly braces, and apostrophes
            formatted_emotion_data = []
            for row in emotion_data:
                id, date, emotion_probability = row
                formatted_date = date.date()  # Remove the time component from the date
                formatted_emotion_probability = emotion_probability.replace("{", "").replace("}", "").replace("'", "").replace('"', '')
                formatted_emotion_data.append((id, formatted_date, formatted_emotion_probability))

            return render_template('monitor_emotions.html', emotion_data=formatted_emotion_data, userMessages=user_messages, username=username)
    
    flash("Please log in to monitor your emotions.")
    return redirect(url_for('login'))


from emotion import predict_emotion_list



@app.route('/mood_tracker', methods=['GET'])
def mood_tracker():
    # Check if 'user_id' is in session
    if 'user_id' in session:
        user_id = session['user_id']

        # Fetch the username associated with the user_id
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            username = user[0]

            # Get messages from request
            messages = request.args.get('messages')
            if messages:
                messages = json.loads(messages)
                emotion_probabilities = predict_emotion_list(messages)
            else:
                emotion_probabilities = {}

            # Get today's date
            today_date = datetime.date.today()

            # Check if there's already an entry for today
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT emotion_probability FROM emotion_probabilities WHERE username=%s AND date=%s", (username, today_date))
            existing_record = cursor.fetchone()

            if existing_record:
                # Calculate the average with the existing record
                existing_probabilities = json.loads(existing_record[0])
                updated_probabilities = {}

                for emotion in emotion_probabilities:
                    if emotion in existing_probabilities:
                        updated_probabilities[emotion] = (emotion_probabilities[emotion] + existing_probabilities[emotion]) / 2
                    else:
                        updated_probabilities[emotion] = emotion_probabilities[emotion]

                for emotion in existing_probabilities:
                    if emotion not in updated_probabilities:
                        updated_probabilities[emotion] = existing_probabilities[emotion]

                # Update the record with the averaged probabilities
                cursor.execute("UPDATE emotion_probabilities SET emotion_probability=%s WHERE username=%s AND date=%s",
                               (json.dumps(updated_probabilities), username, today_date))
            else:
                # Insert a new record if no entry for today exists
                cursor.execute("INSERT INTO emotion_probabilities (username, date, emotion_probability) VALUES (%s, %s, %s)",
                               (username, today_date, json.dumps(emotion_probabilities)))

            mysql.connection.commit()
            cursor.close()

            return render_template('mood_tracker.html', emotion_probabilities=emotion_probabilities, username=username)

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

        # Get the username associated with the user_id
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user_username = cursor.fetchone()
        
        if user_username:
            user_username = user_username[0]
            
            # Delete user data from multiple tables
            cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
            cursor.execute("DELETE FROM emotion_probabilities WHERE username=%s", (user_username,))
            mysql.connection.commit()
            cursor.close()

            session.pop('user_id', None)
            flash("Your account has been deleted successfully.")

            return jsonify({'status': 'success'}), 200
        else:
            cursor.close()
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

@app.route('/initiate_conversation', methods=['GET'])
def initiate_conversation():
    global conversation_started
    conversation_started = True
    response = "Hello, how's your day so far?"
    return jsonify({'response': response})

@app.route('/get_emotion_data', methods=['GET'])
def get_emotion_data():
    if 'user_id' in session:
        user_id = session['user_id']
        
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        user_username = cursor.fetchone()[0]
        cursor.close()
        
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT date, emotion_probability FROM emotion_probabilities WHERE username=%s ORDER BY date DESC LIMIT 5", (user_username,))
        emotion_data = cursor.fetchall()
        cursor.close()
        
        formatted_emotion_data = []
        for row in emotion_data:
            date, emotion_probability = row
            emotion_probability = json.loads(emotion_probability)
            positive = sum([value for key, value in emotion_probability.items() if key in ['joy', 'surprise']])
            negative = sum([value for key, value in emotion_probability.items() if key not in ['joy', 'surprise']])
            total = positive + negative
            
            if total > 0:
                positive1 = (positive / total) * 100
                negative1 = (negative / total) * 100
                formatted_emotion_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'positive': positive1,
                    'negative': negative1
                })
        
        formatted_emotion_data.reverse()  # To show the oldest date first
        return jsonify(formatted_emotion_data)
    
    return jsonify([]), 401


if __name__ == '__main__':
    app.run(debug=True)
