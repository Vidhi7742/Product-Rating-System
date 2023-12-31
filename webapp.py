from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re

app = Flask(__name__)

# Load the dataset
dataa = pd.read_csv('product.csv')
english_stops = set(stopwords.words('english'))

def load_dataset():
    df = pd.read_csv('product.csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    x_data = x_data.replace({'<.*?>': ''}, regex=True)          # remove html tag
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex=True)     # remove non-alphabet
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower() for w in review])   # lower case

    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data

x_data, y_data = load_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))

max_length = get_max_length()

# Load the model and tokenizer
loaded_model = load_model('models/LSTM.h5')
token = Tokenizer(lower=False)
token.fit_on_texts(x_train)

def preprocess_input(review):
    regex = re.compile(r'[^a-zA-Z\s]')
    review = regex.sub('', review)
    words = review.split(' ')
    filtered = [w for w in words if w not in english_stops]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]
    tokenize_words = token.texts_to_sequences(filtered)
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
    return tokenize_words

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        tokenized_review = preprocess_input(review)
        tokenized_review = pad_sequences(tokenized_review, maxlen=max_length, padding='post', truncating='post')
        result = loaded_model.predict(tokenized_review)

        probability_positive_class = float(result[0])  # Extract the probability for the positive class

        if probability_positive_class >= 0.8:
            prediction = '*****'
        elif probability_positive_class > 0.65:
            prediction = '****'
        elif probability_positive_class > 0.5:
            prediction = '***'
        elif probability_positive_class > 0.3:
            prediction = '**'
        else:
            prediction = '*'

        return render_template('index.html', prediction=prediction, review=review)

if __name__ == '__main__':
    app.run(debug=True)
