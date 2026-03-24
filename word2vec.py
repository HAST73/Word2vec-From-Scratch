import random
import numpy as np
import re


# Data preprocessing
class TextData:
    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

        self._build_vocab()

    # Cleans the text, tokenizes it, and builds a word to ID mapping
    def _build_vocab(self):
        # Convert to lowercase to treat "Sherlock" and "sherlock" as the same word
        original_text = self.text.lower()

        # Remove all punctuation and numbers to keep only letters and spaces
        original_text = re.sub(r'[^a-z\s]', '', original_text)

        # Split text into a list of individual words (tokens)
        self.tokens = original_text.split()

        # Get unique words to build the dictionary
        unique_words = set(self.tokens)

        for i, word in enumerate(unique_words):
            self.word2idx[word] = i
            self.idx2word[i] = word

        self.vocab_size = len(self.word2idx)

    # Generates (context_word_id, center_word_id) pairs using a sliding window
    def get_training_pairs(self, window_size):

        pairs = []

        for i, word in enumerate(self.tokens):
            center_word_id = self.word2idx[word]

            # Define window boundaries to avoid going out of list index
            start = max(0, i - window_size)
            end = min(len(self.tokens), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word_id = self.word2idx[self.tokens[j]]
                    pairs.append((context_word_id, center_word_id))

        return pairs


# Word2Vec model implementation using Skip-gram with Negative Sampling (SGNS)
class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # W1: Target/Center word embeddings
        self.W1 = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))

        # W2: Context word embeddings
        self.W2 = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))

    # Numerically stable sigmoid function
    # Prevents overflow errors (NaN) when exp() gets very large positive or negative numbers
    def sigmoid(self, x):

        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),  # Standard formula for positive x
            np.exp(x) / (1 + np.exp(x))  # Equivalent formula for negative x
        )

    # Randomly samples K negative words, ensuring the true context word is excluded
    def get_negative_samples(self, context_id, num_samples):

        negative_samples = []

        while len(negative_samples) < num_samples:
            rand_id = random.randint(0, self.vocab_size - 1)
            if rand_id != context_id:
                negative_samples.append(rand_id)

        return negative_samples

    # Performs a single training step (forward & backward pass) for one word pair
    def train_step(self, center_id, context_id, num_negative_samples):

        # Grab the vectors for our specific words from the matrices
        v_c = self.W1[center_id]
        u_o = self.W2[context_id]

        neg_ids = self.get_negative_samples(context_id, num_negative_samples)
        u_k = self.W2[neg_ids]

        # Forward pass: calculate the dot products for the positive and negative samples
        z_pos = u_o.dot(v_c)
        z_neg = u_k.dot(v_c)

        # Calculate the loss using the formulas from the Stanford lecture notes
        loss_pos = -np.log(self.sigmoid(z_pos))
        # We add eps to prevent log(0) which would give us -inf and break our training
        loss_neg = -np.sum(np.log(self.sigmoid(-z_neg)))
        step_loss = loss_pos + loss_neg

        # Backward pass: calculate gradients and update weights
        force_pos = self.sigmoid(z_pos) - 1
        force_neg = self.sigmoid(z_neg)

        self.W2[context_id] -= self.learning_rate * (force_pos * v_c)
        self.W2[neg_ids] -= self.learning_rate * (np.outer(force_neg, v_c))

        grad_vc = (force_pos * u_o) + np.sum(force_neg[:, np.newaxis] * u_k, axis=0)
        self.W1[center_id] -= self.learning_rate * grad_vc

        # Return the loss for this step so we can track it if we want
        return step_loss


# Evaluation utilities to find most similar and least similar words based on cosine similarity of their embeddings
# Utility to extract the trained vector for a given word
def get_vector(word, data_obj, model_obj):
    word_id = data_obj.word2idx.get(word)
    if word_id is None:
        return None
    return model_obj.W1[word_id]


# Calculates the cosine similarity (angle) between two vectors
def cosine_sim(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Prevent division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


# Returns the top_k words that are closest to the target_word in the vector space
def get_most_similar(target_word, data_obj, model_obj, top_k):
    target_vec = get_vector(target_word, data_obj, model_obj)

    if target_vec is None:
        return [("Word not in vocabulary", 0.0)]

    similarities = []

    for word, word_id in data_obj.word2idx.items():
        if word != target_word:
            other_vec = model_obj.W1[word_id]
            sim = cosine_sim(target_vec, other_vec)
            similarities.append((word, sim))

    # Sort descending (highest similarity first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


# Returns the top_k words that were pushed the furthest away from the target_word
def get_least_similar(target_word, data_obj, model_obj, top_k):
    target_vec = get_vector(target_word, data_obj, model_obj)

    if target_vec is None:
        return [("Word not in vocabulary", 0.0)]

    similarities = []

    for word, word_id in data_obj.word2idx.items():
        if word != target_word:
            other_vec = model_obj.W1[word_id]
            sim = cosine_sim(target_vec, other_vec)
            similarities.append((word, sim))

    # # Sort ascending (lowest/most negative similarity first)
    similarities.sort(key=lambda x: x[1])

    return similarities[:top_k]


# Training loop and evaluation examples
if __name__ == "__main__":
    # Load your custom text dataset
    with open("text_example.txt", "rt", encoding="utf-8") as read_file:
        sample_text = read_file.read()

    print("1. Preparing data...")
    data = TextData(sample_text)
    training_pairs = data.get_training_pairs(window_size=6)
    print(f"Generated {len(training_pairs)} training pairs.\n")

    print("2. Initializing Word2Vec model...")
    model = Word2Vec(vocab_size=data.vocab_size, embedding_dim=50, learning_rate=0.01)

    epochs = 100
    print(f"3. Starting training loop ({epochs} epochs)...\n")

    for epoch in range(epochs):
        # Variable to accumulate the loss for the entire epoch
        epoch_loss = 0.0

        for context_id, center_id in training_pairs:
            # Accumulate loss from each training step
            step_loss = model.train_step(center_id, context_id, num_negative_samples=3)
            epoch_loss += step_loss

        # Print the average loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(training_pairs)
            print(f" - Completed epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    print("Training completed!\n")

    # Testing the results
    print("\n----- Test 1: Nearest neighbors -----")
    test_word = "sherlock"
    print("Most similar to:", test_word)

    results_similar = get_most_similar(test_word, data, model, top_k=5)
    for w, sim in results_similar:
        print(w, round(sim, 4))

    print("\n----- Test 2: Furthest words -----")
    print("Least similar to:", test_word)

    results_opposite = get_least_similar(test_word, data, model, top_k=5)
    for w, sim in results_opposite:
        print(w, round(sim, 4))