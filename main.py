import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Concatenate, Dropout, GlobalMaxPooling1D, Attention, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset")
df = pd.read_csv('train.csv')

def preprocess_text(text):
    return text.str.lower().str.replace('[^\w\s]', '', regex=True)

df['prompt'] = preprocess_text(df['prompt'])
df['response_a'] = preprocess_text(df['response_a'])
df['response_b'] = preprocess_text(df['response_b'])

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 250

print("Creating tokenizer")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df['prompt'].tolist() + df['response_a'].tolist() + df['response_b'].tolist())

X_prompt = tokenizer.texts_to_sequences(df['prompt'])
X_response_a = tokenizer.texts_to_sequences(df['response_a'])
X_response_b = tokenizer.texts_to_sequences(df['response_b'])

X_prompt = pad_sequences(X_prompt, maxlen=MAX_SEQUENCE_LENGTH)
X_response_a = pad_sequences(X_response_a, maxlen=MAX_SEQUENCE_LENGTH)
X_response_b = pad_sequences(X_response_b, maxlen=MAX_SEQUENCE_LENGTH)

print("Encoding features")
le_model = LabelEncoder()
df['model_a_encoded'] = le_model.fit_transform(df['model_a'])
df['model_b_encoded'] = le_model.transform(df['model_b'])

X_model_a = df['model_a_encoded'].values
X_model_b = df['model_b_encoded'].values

y = df[['winner_model_a', 'winner_model_b', 'winner_tie']].values
y = np.argmax(y, axis=1)

print("Train test split")
X_train_prompt, X_test_prompt, X_train_response_a, X_test_response_a, X_train_response_b, X_test_response_b, \
X_train_model_a, X_test_model_a, X_train_model_b, X_test_model_b, y_train, y_test = train_test_split(
    X_prompt, X_response_a, X_response_b, X_model_a, X_model_b, y, test_size=0.2, random_state=42)

print("Computing weight class")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

print("Pretraining GloVe")
EMBEDDING_DIM = 300
embedding_index = {}
with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

print("Embedded matrix")
word_index = tokenizer.word_index
embedding_matrix = np.zeros((MAX_VOCAB_SIZE, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print("Input encoding")
def build_attention_lstm(input_name):
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), name=input_name)
    embedding_layer = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, 
                                 weights=[embedding_matrix], trainable=False)(input_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    attention_layer = Attention()([lstm_layer, lstm_layer])
    global_max_pooling = GlobalMaxPooling1D()(attention_layer)
    return input_layer, global_max_pooling

input_prompt, lstm_prompt = build_attention_lstm('prompt_input')
input_response_a, lstm_response_a = build_attention_lstm('response_a_input')
input_response_b, lstm_response_b = build_attention_lstm('response_b_input')

input_model_a = Input(shape=(1,), name='model_a_input')
embedding_model_a = Embedding(input_dim=len(le_model.classes_), output_dim=16)(input_model_a)
flat_model_a = Flatten()(embedding_model_a)

input_model_b = Input(shape=(1,), name='model_b_input')
embedding_model_b = Embedding(input_dim=len(le_model.classes_), output_dim=16)(input_model_b)
flat_model_b = Flatten()(embedding_model_b)

print("Concatening features")
concatenated = Concatenate()([lstm_prompt, lstm_response_a, lstm_response_b, flat_model_a, flat_model_b])
dense1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(concatenated)
dropout = Dropout(0.5)(dense1)
output = Dense(3, activation='softmax')(dropout)

print("Compiling the Model")
model = Model(inputs=[input_prompt, input_response_a, input_response_b, input_model_a, input_model_b], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training the Model")
history = model.fit([X_train_prompt, X_train_response_a, X_train_response_b, X_train_model_a, X_train_model_b], 
                    y_train, epochs=15, batch_size=64, validation_split=0.1, class_weight=class_weights)

print("Evaluation")
loss, accuracy = model.evaluate([X_test_prompt, X_test_response_a, X_test_response_b, X_test_model_a, X_test_model_b], y_test)
print(f'Test Accuracy: {accuracy:.2f}')

y_pred = np.argmax(model.predict([X_test_prompt, X_test_response_a, X_test_response_b, X_test_model_a, X_test_model_b]), axis=1)
print(classification_report(y_test, y_pred, target_names=['Winner Model A', 'Winner Model B', 'Tie']))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Winner A', 'Winner B', 'Tie'], 
            yticklabels=['Winner A', 'Winner B', 'Tie'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
