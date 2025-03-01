import numpy as np
from collections import Counter
from Generative_Model import *
from Learning import *
from CG1 import *
from chunks import *
import PIL as PIL
from PIL import Image
import os
from time import time
from chunks import *

def calculate_sem(data):
    """
    Calculate the standard error of the mean (SEM) for a given array of data.

    Parameters:
        data (numpy.ndarray): A numpy array containing the data points.

    Returns:
        float: The standard error of the mean of the data.
    """
    # Calculate the standard deviation of the data
    std_dev = np.std(data, ddof=1)  # ddof=1 provides an unbiased estimator by using N-1 in the denominator

    # Calculate the number of observations in the data
    n = len(data)

    # Calculate SEM
    sem = std_dev / np.sqrt(n)

    return sem


def lzcompression(array_string):
    # input: string, output: lz evaluation
    def lz_complexity(sequence):
        seql = 0 # length of sequence after compression
        parsed_sequence = []
        n = len(sequence)
        phrases = dict()
        i = 0

        while i < n:
            j = i + 1
            while j <= n:
                current_phrase = sequence[i:j]
                if tuple(current_phrase) not in phrases:
                    phrases[tuple(current_phrase)] = 1
                    break
                else:
                    phrases[tuple(current_phrase)] += 1
                j += 1

            parsed_sequence.append(tuple(current_phrase))
            seql = seql + 1  # increment size of the parsed sequence

            i = j
        return phrases, seql, parsed_sequence

    # Convert seq as np array into string:
    sequence = array_string
    phrases, seql, parsed_sequence = lz_complexity(sequence)
    count_freq = Counter(parsed_sequence) # how often each word is being parsed
    freq = np.array(list(count_freq.values()))
    complexity = 0
    for k in count_freq:
        count_freq[k] = count_freq[k]/freq.sum()
    ps = freq / freq.sum()
    storage = -np.sum(np.log2(ps))# storage cost of all chunks
    for k in parsed_sequence:
        complexity = complexity - np.log2(count_freq[k])
    # print(f"Sequence Complexity by LZ: {complexity} bits")
    return complexity, seql, storage

############################# coding efficiency
def lz_complexity_coding_efficiency(sequence):
    n_entries_dict = []  # number of entries in the dictionary (excluding overhead)
    parsed_seql_record = []  # progress in encoding the length of the sequence

    seql = 0  # length of sequence after compression
    parsed_sequence = []
    n = len(sequence)
    phrases = dict()
    i = 0
    while i < n:
        j = i + 1
        while j <= n:
            current_phrase = sequence[i:j]
            if tuple(current_phrase) not in phrases:
                phrases[tuple(current_phrase)] = 1
                break
            else:
                phrases[tuple(current_phrase)] += 1
            j += 1

        parsed_sequence.append(tuple(current_phrase))
        seql = seql + 1  # increment size of the parsed sequence
        n_entries_dict.append(seql)
        parsed_seql_record.append(i)

        i = j
    return n_entries_dict, parsed_seql_record



def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



def tokenize(text):
    # Basic tokenization to split the text into words
    # tokens = re.findall(r'\b\w+\b', text) # words
    tokens = [char for char in text] # characters
    return tokens


def build_vocabulary(tokens, min_freq=1):
    # Count the occurrence of each word in the dataset
    word_counts = Counter(tokens)

    # Build the vocabulary by including words that appear
    # at least min_freq times
    vocabulary = {word: i for i, (word, freq) in enumerate(word_counts.items()) if freq >= min_freq}

    # Add special tokens
    vocabulary['<PAD>'] = len(vocabulary)
    vocabulary['<UNK>'] = len(vocabulary)

    return vocabulary

def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read()
    return text


plot = False

# load data
# train on chunking model
# evaluate preplexity

import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Script that receives BabyLM's filename")

# Add arguments
parser.add_argument("-n", "--filename", type=str, help="Filename", default="bnc_spoken.train")

# Parse the arguments
args = parser.parse_args()

train_path = './train_10M/' + args.filename

train_text = read_text_file(train_path)

# Tokenize the text
train_tokens = tokenize(train_text)
# Build the vocabulary
vocab = build_vocabulary(train_tokens)
train_tokens = [vocab[w] for w in train_tokens]
# Print some information about the dataset and vocabulary
print(f"Number of tokens in training data: {len(train_tokens)}")
print(f"Size of vocabulary: {len(vocab)}")
print("Some example words in vocabulary:", list(vocab.keys())[:20])

# TODO: iterations across samples here:
seqL = 1000
n_run = 10
fullseq = train_text[:seqL*n_run]
slice_sz = 1000
n_measure = 1  # different model evaluation measures. For LZ, it just measures the sequence complexity
n_iter = 10 # iteration corresponding to the chunking and variable learning models, surrogate measure that is meant to map to the number of iterations in the cognitve models
datalz_length = np.empty((n_run, n_iter, n_measure))
sequence_original_length = np.empty((n_run, n_iter, n_measure))
datalz_complexity = np.empty((n_run, n_iter, n_measure))
datalz_storage = np.empty((n_run, n_iter, n_measure))

i = 0  # in each iteration, use the same data for training 14 number of epoches
for seq in slicer(fullseq, slice_sz):  # the same sequence as in
    # lz compression complexity (about constant)
    complexity, seql, storage = lzcompression(seq)
    print('seql after compression ', seql)
    datalz_length[i, :, :] = np.array(seql)
    datalz_complexity[i, :, :] = np.array(complexity)
    datalz_storage[i, :, :] = np.array(storage)
    sequence_original_length[i, :, :] = np.array(len(seq))
    i = i + 1

np.save(f'./data/babyspeech/{args.filename}_lz_seql.npy', datalz_length)
np.save(f'./data/babyspeech/{args.filename}_lz_complexity.npy', datalz_complexity)
np.save(f'./data/babyspeech/{args.filename}_lz_storage.npy', datalz_storage)
np.save(f'./data/babyspeech/{args.filename}_sequence_original_length.npy', sequence_original_length)

#################################### Now the hierarchical learning models ############################
fullseq = np.array(train_tokens).reshape([len(train_tokens), 1, 1])

slice_sz = 1000
n_measure = 9
n_run = 10
n_iter = 25
datahcm = np.empty((n_run, n_iter, n_measure))
datahvm = np.empty((n_run, n_iter, n_measure))
i = 0
for seq in slicer(fullseq, slice_sz):
    if i == n_run:break
    cghvm = CG1(DT=0.1, theta=0.996)
    cghvm = hcm_markov_control_1(seq, cghvm, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    datahcm[i,:,:] = np.array(cghcm.learning_data)
    datahvm[i,:,:] = np.array(cghvm.learning_data)
    i = i + 1
np.save(f'./data/babyspeech/{args.filename}_hcm.npy', datahcm)
np.save(f'./data/babyspeech/{args.filename}_hvm.npy', datahvm)

import matplotlib.pyplot as plt
import numpy as np

# plot the cognitive model learning curve, how it the iteration changes

if plot:
    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables', 'storage cost']

    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(datahcm[0, :, 0])

    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        hcm_mean = np.mean(datahcm[:, :, i + 1], axis=0)
        hvm_mean = np.mean(datahvm[:, :, i + 1], axis=0)
        ax.plot(x, hcm_mean, label='HCM', color='orange', linewidth=4, alpha=0.3)
        ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha=0.3)
        for j in range(0, datahcm.shape[0]):
            ax.plot(x, datahcm[j, :, i + 1], color='orange', linewidth=1, alpha=0.3)
            ax.plot(x, datahvm[j, :, i + 1], color='blue', linewidth=1, alpha=0.3)

        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig(f'./data/babyspeech/{args.filename}_HCM_HVM_learning_progress_comparison.png')


#################################### Compare HVM, HCM, and LZ78 on coding efficiency #################################
# report as table in the updated evalution measure
seql = 1000

hcm_min_seq_l = seql / np.max(datahcm[:, -1, 3], axis=0)  # at the end of training
hvm_min_seq_l = seql / np.max(datahvm[:, -1, 3], axis=0)  # average over different runs
lz_min_seq_l = np.max(datalz_length[:, -1, 0], axis=0)   # average over different runs

hcm_min_complexity = np.min(datahcm[:, -1, 4], axis=0)  # average over different runs
hvm_min_complexity = np.min(datahvm[:, -1, 4], axis=0)  # average over different runs
lz_min_complexity = np.min(datalz_complexity[:, -1, 0], axis=0)  # average over different runs

#############
if plot:
    plt.rcParams['font.size'] = 18
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 3 rows, 1 column

    # Sequence Length
    models = ['HCM', 'HVM', 'LZ78']
    seq_length = [hcm_mean_seq_l, hvm_mean_seq_l, lz_mean_seq_l]
    axes[0].bar(models, seq_length, color=['#CC5500', 'royalblue', '#36454F'], edgecolor='black',
                yerr=sem_seq_l)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Parsing Length |W|')
    plt.yscale('linear')

    # Sequence Complexity
    models = ['HCM', 'HVM', 'LZ78']
    seq_complexity = [hcm_mean_complexity, hvm_mean_complexity, lz_mean_complexity]
    axes[1].bar(models, seq_complexity, color=['#CC5500', 'royalblue', '#36454F'], edgecolor='black',
                yerr=sem_complexity)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Sequence Likelihood -logP(S)')
    plt.yscale('linear')



# Flatten the array to 1D
# Convert the 1D array to a string of characters
overhead_char = 0
seql = 1000
n_run = 10
fullseq = train_text[:seql*n_run]
slice_sz = 1000
n_measure = 1  # different model evaluation measures. For LZ, it just measures measure the sequence complexity
n_iter = 10 # iteration corresponding to the chunking and variable learning models, surrogate measure that is meant to map to the number of iterations in the cognitve models
datalz_compression_efficiency = np.empty((n_run, n_iter, n_measure))

i = 0  # in each iteration, use the same data for training 14 number of epoches
for seq in slicer(fullseq, slice_sz):  # the same sequence as in
    # lz compression complexity (about constant)
    n_entries_dict, parsed_seql_record = lz_complexity_coding_efficiency(seq)
    n_entries_dict = np.array(n_entries_dict) + overhead_char
    parsed_seql_record = np.array(parsed_seql_record)
    lzcodingefficiency = n_entries_dict / parsed_seql_record
    # only evaluate the coding efficnecy at the end of sequence length 1000
    datalz_compression_efficiency[i,:,:] = lzcodingefficiency[-1]
    i = i + 1


hcm_coding_efficiency = datahcm[:, -1, 6]/seql # this would be an array over different runs
hvm_coding_efficiency = datahvm[:, -1, 6]/seql

hcm_min_coding_efficiency = np.min(hcm_coding_efficiency)  # at the end of training
hvm_min_coding_efficiency = np.min(hvm_coding_efficiency)  # average over different runs
lz_min_coding_efficiency = np.min(datalz_compression_efficiency[:, -1, 0])  # average over different runs

print(f'seql: min [hcm, hvm, lz] {hcm_min_seq_l:.2f}, {hvm_min_seq_l:.2f}, {lz_min_seq_l:.2f}')
print(f'complexity: min [hcm, hvm, lz] {hcm_min_complexity:.2f},{hvm_min_complexity:.2f},{lz_min_complexity:.2f}')
print(f'coding efficiency: min [hcm, hvm, lz] {hcm_min_coding_efficiency:.2f},{hvm_min_coding_efficiency:.2f},{lz_min_coding_efficiency:.2f}')
