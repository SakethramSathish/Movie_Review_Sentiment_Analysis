from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np

class SentimentModel:
    def __init__(self, max_words=10000, max_len=100):
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.max_words = max_words
        self.max_len = max_len
        self.model = None

    def tokenize_and_pad(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return padded

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len),
            Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, kernel_regularizer=l2(0.005))),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
            Dropout(0.4),
            Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        self.model = model
        return self.model