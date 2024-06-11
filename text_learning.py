import re
from collections import Counter
import numpy as np

def read_wikitext_file(filepath):
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read()
    return text

def tokenize(text):
    # Basic tokenization to split the text into words
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_vocabulary(tokens, min_freq=5):
    # Count the occurrence of each word in the dataset
    word_counts = Counter(tokens)

    # Build the vocabulary by including words that appear
    # at least min_freq times
    vocabulary = {word: i for i, (word, freq) in enumerate(word_counts.items()) if freq >= min_freq}

    # Add special tokens
    vocabulary['<PAD>'] = len(vocabulary)
    vocabulary['<UNK>'] = len(vocabulary)

    return vocabulary


from collections import Counter, defaultdict


def train_bigram_model(tokens):
    bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))
    word_counts = Counter(tokens)

    # Compute the probability of a word given the previous word
    model = defaultdict(Counter)
    for (word1, word2), bigram_count in bigram_counts.items():
        word1_count = word_counts[word1]
        model[word1][word2] = bigram_count / word1_count

    return model

def generate_sequence_wikitext2():
    # Assuming the dataset is downloaded and the file paths are known
    train_path = '../wikitext-2/wiki.train.tokens'
    valid_path = '../wikitext-2/wiki.valid.tokens'
    test_path = '../wikitext-2/wiki.test.tokens'

    train_text = read_wikitext_file(train_path)
    valid_text = read_wikitext_file(valid_path)
    test_text = read_wikitext_file(test_path)

    # Tokenize the text
    train_tokens = tokenize(train_text)

    # Build the vocabulary
    vocab = build_vocabulary(train_tokens)

    # Print some information about the dataset and vocabulary
    print(f"Number of tokens in training data: {len(train_tokens)}")
    print(f"Size of vocabulary: {len(vocab)}")
    print("Some example words in vocabulary:", list(vocab.keys())[:20])

    # process the train tokens into training sequences
    seq_train = [vocab.get(token, vocab['<UNK>']) for token in train_tokens]

    seq_train = np.array(seq_train).reshape((len(seq_train), 1, 1))

    # if needed, can combine the sequence training and test together.
    # # Tokenize the test data
    # Train the bigram model
    bigram_model = train_bigram_model(train_tokens)
    test_tokens = tokenize(test_text)

    # Compute the perplexity of the bigram model on the test data
    perplexity = compute_perplexity(bigram_model, test_tokens)
    # Print the perplexity
    print(f'Perplexity of the bigram model on the test data: {perplexity:.2f}')
    return seq_train



import math


def compute_perplexity(model, tokens):
    N = len(tokens)
    log_prob = 0

    for i in range(1, N):
        word1 = tokens[i - 1]
        word2 = tokens[i]
        # Use a small value for smoothing (handling unknown bigrams)
        prob = model[word1].get(word2, 1e-6)
        log_prob += math.log(prob)

    # Compute perplexity
    perplexity = math.exp(-log_prob / N)

    return perplexity




