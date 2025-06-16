import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTMMemorySelector:
    def __init__(self, vocab_size=2000, maxlen=20, embedding_dim=64, lstm_units=32):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.model = self._build_model()
        self.is_fit = False

    def _build_model(self):
        mem_input = Input(shape=(self.maxlen,))
        query_input = Input(shape=(self.maxlen,))
        embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)
        mem_emb = embedding(mem_input)
        query_emb = embedding(query_input)
        lstm = LSTM(self.lstm_units)
        mem_vec = lstm(mem_emb)
        query_vec = lstm(query_emb)
        merged = Concatenate()([mem_vec, query_vec])
        dense = Dense(32, activation='relu')(merged)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model([mem_input, query_input], output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit_tokenizer(self, memory_texts, queries=None):
        texts = list(memory_texts)
        if queries is not None:
            texts += list(queries)
        self.tokenizer.fit_on_texts(texts)
        self.is_fit = True

    def encode(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=self.maxlen)

    def train(self, memory_texts, queries, labels, epochs=10, batch_size=8):
        if not self.is_fit:
            self.fit_tokenizer(memory_texts, queries)
        mem_seq = self.encode(memory_texts)
        query_seq = self.encode(queries)
        self.model.fit([mem_seq, query_seq], np.array(labels), epochs=epochs, batch_size=batch_size)

    def score_memories(self, memory_texts, query):
        if not self.is_fit:
            self.fit_tokenizer(memory_texts, [query])
        mem_seq = self.encode(memory_texts)
        query_seq = np.repeat(self.encode([query]), len(memory_texts), axis=0)
        scores = self.model.predict([mem_seq, query_seq])
        return scores.flatten()

    def select_best_memory(self, memory_texts, query):
        scores = self.score_memories(memory_texts, query)
        best_idx = np.argmax(scores)
        return memory_texts[best_idx], scores[best_idx]


if __name__ == '__main__':
    # Sample data
    memory_texts = [
        "the cat sat on the mat",
        "the dog ate my homework",
        "the sky is blue",
        "the sun is bright"
    ]
    queries = [
        "what did the cat do?",
        "what happened to my homework?",
        "what color is the sky?",
        "how is the sun?"
    ]
    # In a real scenario, labels would be 0 or 1, indicating relevance.
    # Here, we'll just create some dummy labels for demonstration.
    # Let's assume the queries are relevant to the memories at the same index.
    labels = [1, 1, 1, 1]

    # Initialize and train the selector
    selector = LSTMMemorySelector(vocab_size=100, maxlen=10, embedding_dim=8, lstm_units=8)
    print("Training the model...")
    selector.train(memory_texts, queries, labels, epochs=20)
    print("Training complete.")

    # --- Test scoring ---
    test_query = "what is the color of the sky?"
    print(f"\nScoring memories for query: '{test_query}'")
    scores = selector.score_memories(memory_texts, test_query)
    for mem, score in zip(memory_texts, scores):
        print(f"  Memory: '{mem}', Score: {score:.4f}")

    # --- Test selection ---
    test_query_2 = "tell me about the dog"
    print(f"\nSelecting best memory for query: '{test_query_2}'")
    best_memory, best_score = selector.select_best_memory(memory_texts, test_query_2)
    print(f"  Best Memory: '{best_memory}', Score: {best_score:.4f}")

    # --- Test another selection ---
    test_query_3 = "what was on the mat?"
    print(f"\nSelecting best memory for query: '{test_query_3}'")
    best_memory_3, best_score_3 = selector.select_best_memory(memory_texts, test_query_3)
    print(f"  Best Memory: '{best_memory_3}', Score: {best_score_3:.4f}") 