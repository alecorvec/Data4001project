import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def load_data():
    df = pd.read_csv("train.csv")
    return df

def train_network(df):
    
    df["target"] = df.apply(lambda x: 0 if x['winner_model_a'] == 1 else (1 if x['winner_model_b'] == 1 else 2), axis=1)
    df.pop("winner_model_a")
    df.pop("winner_model_b")
    df.pop("winner_tie")
    df_y = df.pop("target")
    df_x = np.array(df)

    model = ks.models.Sequential([
        ks.layers.Dense(64),
        ks.layers.Dense(1)
    ])

    model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())
    
    model.fit(df_x, df_y, epochs=10)

    
    # df['target'] = df.apply(lambda x: 0 if x['winner_model_a'] == 1 else (1 if x['winner_model_b'] == 1 else 2), axis=1)

    # X = 0
    # y = df['target']

    # max_features = 10000
    # sequence_length = 250

    # vectorize_layer = ks.layers.TextVectorization(
    #     max_tokens=max_features,
    #     output_mode='int',
    #     output_sequence_length=sequence_length)

    # train_text = df.map(lambda x: x)
    # vectorize_layer.adapt(train_text)
    
    
    # embedding_dim = 16

    # model = tf.keras.Sequential([
    #     ks.layers.Embedding(max_features + 1, embedding_dim),
    #     ks.layers.Dropout(0.2),
    #     ks.layers.GlobalAveragePooling1D(),
    #     ks.layers.Dropout(0.2),
    #     ks.layers.Dense(1)])

    # model.summary()



    # model = ks.models.Sequential()
    # model.add(ks.layers.Dense(12, input_shape=(8,), activation='relu'))
    # model.add(ks.layers.Dense(8, activation='relu'))
    # model.add(ks.layers.Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X, y, epochs=150, batch_size=10)
    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))

    return 0

def predict():
    return 0

if __name__ == "__main__" :
    print("tensorflow:", tf.__version__)
    print("keras:", ks.__version__)
    df = load_data()
    #if (df):
    train_network(df=df)
    exit(0)
    #else:
    #    exit(84)
    