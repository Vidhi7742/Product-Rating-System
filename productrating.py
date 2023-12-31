import pandas as pd    # to load dataset
import numpy as np     # for mathematic equation
from nltk.corpus import stopwords   # to get collection of stopwords
from sklearn.model_selection import train_test_split # for splitting dataset
# load saved model
import re
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.layers import Embedding, LSTM, Dense # layers of the architecture
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from tensorflow.keras.models import load_model 

dataa = pd.read_csv('product.csv')

print(dataa)

english_stops = set(stopwords.words('english'))


def load_dataset():
    df = pd.read_csv('product.csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    x_data = x_data.replace({'<.*?>': ''}, regex = True)          # remove html tag
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)     # remove non alphabet
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower() for w in review])   # lower case
    
    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data

x_data, y_data = load_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

# print('Train Set')
# print('Test Set')
# print(y_train, '\n')
# print(y_test)

def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))

token = Tokenizer(lower=False)    # no need lower, because already lowered the data in load_data()
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1   # add 1 because of 0 padding

# print('Encoded X Train\n', x_train, '\n')
# print('Encoded X Test\n', x_test, '\n')
# print('Maximum review length: ', max_length)


# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length = max_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# print(model.summary())

checkpoint = ModelCheckpoint(
    'models/LSTM.h5',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)

model.fit(x_train, y_train, batch_size = 128, epochs = 5, callbacks=[checkpoint])

loss, accuracy = model.evaluate(x_train, y_train)

# Print accuracy
# print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

loaded_model = load_model('models/LSTM.h5')

review = str(input('Product Review: '))

# Pre-process input
regex = re.compile(r'[^a-zA-Z\s]')
review = regex.sub('', review)
#print('Cleaned: ', review)

words = review.split(' ')
filtered = [w for w in words if w not in english_stops]
filtered = ' '.join(filtered)
filtered = [filtered.lower()]

#print('Filtered: ', filtered)

tokenize_words = token.texts_to_sequences(filtered)
tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
#print(tokenize_words)

result = loaded_model.predict(tokenize_words)
#print(result)

if result >= 0.8:
    print('*****')
elif result>0.65:
    print('****')
elif result>0.5:
    print('***')
elif result>0.3:
    print('**')
else:
    print('*')


