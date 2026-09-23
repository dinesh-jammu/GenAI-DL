import json
import pickle
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

VOCAB_SIZE = 15000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 4

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("Loading IMDb dataset...")
    (x_train_idx, y_train), (x_test_idx, y_test) = imdb.load_data(num_words=None)

    word_index = imdb.get_word_index()
    index_to_word = {v + 3: k for k, v in word_index.items()}

    # Clean decode without artificial special tokens (<start>, <pad>, etc.)
    def decode_clean(seq):
        words = [index_to_word[i] for i in seq if i in index_to_word and i >= 4]
        return " ".join(words)

    print("Decoding raw reviews cleanly without artificial tokens...")
    x_train_text = [decode_clean(seq) for seq in x_train_idx]
    x_test_text = [decode_clean(seq) for seq in x_test_idx]

    print("Cleaning text...")
    x_train_clean = [clean_text(t) for t in x_train_text]
    x_test_clean = [clean_text(t) for t in x_test_text]

    print("Fitting tokenizer...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_clean)

    x_train_seq = tokenizer.texts_to_sequences(x_train_clean)
    x_test_seq = tokenizer.texts_to_sequences(x_test_clean)

    print("Padding sequences with PRE-padding...")
    x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Building Bidirectional LSTM Model...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(LSTM_UNITS // 2, dropout=0.2)),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    print(f"Training model for up to {EPOCHS} epochs...")
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    model.fit(
        x_train_pad,
        y_train,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    print("Evaluating on test set...")
    y_prob = model.predict(x_test_pad, batch_size=BATCH_SIZE).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC : {auc:.4f}")

    print("Saving artifacts...")
    model.save("sentiment_lstm.keras")
    model.save("sentiment_lstm.h5")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "test_accuracy": float(acc),
        "test_auc": float(auc),
        "padding_type": "pre"
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Successfully saved sentiment_lstm.keras, sentiment_lstm.h5, tokenizer.pkl, and config.json!")

if __name__ == "__main__":
    main()
