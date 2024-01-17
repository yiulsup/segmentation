import numpy as np
import tensorflow as tf

class Tokenizer:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

    def forward(self, text):
        embedded_sequence = self.embedding(text)
        return embedded_sequence

class SelfAttention:
    def __init__(self):
        self.softmax_values = None

    def feed_forward(self, values):
        return np.sum(values * self.softmax_values.reshape(-1, 1), axis=0)

    def forward(self, key, query):
        attention_scores = np.dot(key, query)
        attention_scores = attention_scores / np.sqrt(query.shape[-1])

        exp_values = np.exp(attention_scores - np.max(attention_scores))
        self.softmax_values = exp_values / np.sum(exp_values, axis=0)

        return self.feed_forward(key)

# Example usage:
vocab_size = 10000  # replace with your vocabulary size
embed_size = 5   # replace with your embedding size

tokenizer = Tokenizer(vocab_size, embed_size)
text = "Hello world i am kevin "

embedded_sequence = tokenizer.forward(text)

key = np.random.rand(embed_size, embed_size)
self_attention = SelfAttention()

weighted_sum = self_attention.forward(key, embedded_sequence)

print("Weighted Sum:", weighted_sum)



        
