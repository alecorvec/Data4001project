import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the dataset
# Assuming the dataset is loaded into a pandas DataFrame called `df`
df = pd.read_csv('train.csv')

# Define maximum vocab size and sequence length
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200

# Preprocessing text data for the prompt, response_a, and response_b
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df['prompt'].tolist() + df['response_a'].tolist() + df['response_b'].tolist())

# Convert text to sequences
X_prompt = tokenizer.texts_to_sequences(df['prompt'])
X_response_a = tokenizer.texts_to_sequences(df['response_a'])
X_response_b = tokenizer.texts_to_sequences(df['response_b'])

# Pad sequences to ensure equal length
X_prompt = pad_sequences(X_prompt, maxlen=MAX_SEQUENCE_LENGTH)
X_response_a = pad_sequences(X_response_a, maxlen=MAX_SEQUENCE_LENGTH)
X_response_b = pad_sequences(X_response_b, maxlen=MAX_SEQUENCE_LENGTH)

# Define target variable
y = df[['winner_model_a', 'winner_model_b', 'winner_tie']].values
y = np.argmax(y, axis=1)  # Convert to a single label for multi-class classification

# Split data into training and testing sets
X_train_prompt, X_test_prompt, X_train_response_a, X_test_response_a, X_train_response_b, X_test_response_b, y_train, y_test = train_test_split(
    X_prompt, X_response_a, X_response_b, y, test_size=0.2, random_state=42)

# Build the model
embedding_dim = 128

# Prompt input
input_prompt = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_prompt = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=embedding_dim)(input_prompt)
lstm_prompt = LSTM(64)(embedding_prompt)

# Response A input
input_response_a = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_response_a = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=embedding_dim)(input_response_a)
lstm_response_a = LSTM(64)(embedding_response_a)

# Response B input
input_response_b = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_response_b = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=embedding_dim)(input_response_b)
lstm_response_b = LSTM(64)(embedding_response_b)

# Concatenate all features
concatenated = Concatenate()([lstm_prompt, lstm_response_a, lstm_response_b])
dense1 = Dense(128, activation='relu')(concatenated)
dropout = Dropout(0.5)(dense1)
output = Dense(3, activation='softmax')(dropout)  # 3 output classes

# Compile and create model
model = Model(inputs=[input_prompt, input_response_a, input_response_b], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train_prompt, X_train_response_a, X_train_response_b], y_train,
          epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate([X_test_prompt, X_test_response_a, X_test_response_b], y_test)
print(f'Test Accuracy: {accuracy:.2f}')
