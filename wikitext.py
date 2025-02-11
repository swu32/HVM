"""All the text processing datafiles """
from Learning import *
from CG1 import *
from chunks import *
import numpy as np
import PIL as PIL
import os
from time import time
from chunks import *
from collections import Counter
import re

def wikitext():
    # evaluate preplexity on wikitext
    # TODO: train HCM on wikitext's training set,
    # TODO: validate HCM on wikitext's testing set, including the perplexity score
    seq = generate_sequence_wikitext2()

    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_depth_parsing(seq, cghcm)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    return

def test_wikitext_parsing():


    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_depth_parsing(seq, cghcm)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    return

def read_wikitext_file(filepath):
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read()
    return text


def tokenize(text):
    # Basic tokenization to split the text into words
    # tokens = list(text)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def text_rep_learning():
    # load data
    # train on chunking model
    # evaluate preplexity

    # Assuming the dataset is downloaded and the file paths are known
    train_path = './train_10M/childes.train'
    #valid_path = '/Users/swu/Documents/MouseHCM/HSTC/wikitext-2/wiki.valid.txt'
    #test_path = '/Users/swu/Documents/MouseHCM/HSTC/wikitext-2/wiki.test.txt'

    train_text = read_wikitext_file(train_path)
    #valid_text = read_wikitext_file(valid_path)
    #test_text = read_wikitext_file(test_path)

    # Tokenize the text
    train_tokens = tokenize(train_text)
    #valid_tokens = tokenize(valid_text)
    #test_tokens = tokenize(test_text)

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

    def translate_chunks_to_words(contenttuple, decoder):
        return decoder[contenttuple[3]]

    def printchunk(chunk, text_file, level=0):
        for content in chunk.ordered_content:
            if isinstance(content, set):
                for contenttuple in sorted(list(content)):
                    contenttoprint = translate_chunks_to_words(contenttuple, decoder)
                    print("\t" * level + contenttoprint)
                    text_file.write("\t" * level + contenttoprint + '\n')

            else:  # a variable, content type os a string
                for chunk in list(cghvm.variables[content].entailingchunks.values()):
                    printchunk(chunk, text_file, level=level + 1)
        return



    # Build the vocabulary
    vocab = build_vocabulary(train_tokens)
    train_tokens = [vocab[w] for w in train_tokens]
    # Print some information about the dataset and vocabulary
    print(f"Number of tokens in training data: {len(train_tokens)}")
    print(f"Size of vocabulary: {len(vocab)}")
    print("Some example words in vocabulary:", list(vocab.keys())[:20])

    slice_sz = 3000
    fullseq = np.array(train_tokens).reshape([-1, 1, 1])[:slice_sz,:,:]
    # val_data = np.array(corpus.valid).reshape([-1, 1, 1])
    # test_data = np.array(corpus.test).reshape([-1, 1, 1])
    n_measure = 9
    n_iter = int(len(fullseq) / slice_sz)
    datahvm = np.empty((n_iter, n_measure)) # at the end of learning progress
    i = 0
    cghvm = CG1(DT=0.1, theta=0.996)
    for seq in slicer(fullseq, slice_sz):
        cghvm = hcm_markov_control_1(seq, cghvm, ABS=True, MAXit=20)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        datahvm[i, :] = np.array(cghvm.learning_data[-1])
        i = i + 1

    # print out learned chunks
    decoder = {value: key for key, value in vocab.items()}
    text_file = open("Wikitext_abstraction_learning.txt", "w")

    for chunk in list(cghvm.chunks.values()):
        if chunk.T > 1:
            print('****************')
            text_file.write('****************\n')
            printchunk(chunk, text_file, level=0)
    text_file.close()

    plot_model_learning_progress(datahvm, savename='./data/wikitext.png')

    # train, val, and test are 20:1:1
    unit_size = 50
    for i in range(0, 1000):# 100 times the unitsize
        cg = CG1(DT=0.00001, theta=0.99998)  # initialize chunking part with specified parameters
        cg, chunkrecord_train = hcm_learning(train_data[20*unit_size*i: 20*unit_size*(i+1),:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        cg, chunkrecord_test = hcm_learning(test_data[unit_size*i: unit_size*(i+1),:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        perplexity_test = evaluate_perplexity(test_data[unit_size*i: unit_size*(i+1),:,:], chunkrecord_test)
        print('test perplexity is ', perplexity_test)
        cg, chunkrecord_val = hcm_learning(val_data[unit_size*i: unit_size*(i+1),:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        perplexity_val = evaluate_perplexity(val_data[unit_size*i: unit_size*(i+1),:,:], chunkrecord_val)
        print('validation perplexity is ', perplexity_val)
        print(i)
        # should run this code on the cluster, and record the perplexity for testing and validation with the number of epochs,
        # at the moment, infeasible to run this code on PC with limied computing power.
        # also, need to integrate chunk deletion mechanism.
    # evaluate the probability of the test set
    # evaluate the sum of log probability across n observations
    # divide by N
    # remove the log by exponentiating
    return

def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def plot_model_learning_progress(datahvm, savename = 'modelcomparison.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    # both are three dimensional arrays

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables', 'storage cost']

    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(datahcm[0,:, 0])

    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        hvm_mean = np.mean(datahvm[:, :, i + 1], axis = 0)
        ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha = 0.3)
        for j in range(0, datahvm.shape[0]):
            ax.plot(x, datahvm[j, :, i + 1], color='blue', linewidth=1, alpha = 0.3)

        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig(savename)

    return


def learn_hunger_game():
    ############### Learning Demonstration on Real World Datasets ###############
    file1 = list(open('text_data.txt').read())

    file1 = [s.lower() for s in file1]
    # convert the list of characters into list of integers, with empty space being the 0 integer.
    character_counts = Counter(file1)
    unique_char = sorted(character_counts, key=character_counts.get, reverse=True)
    unique_char.insert(0,'lll')
    index_to_char = {index: char for index, char in enumerate(unique_char)}
    char_to_index = {char: index for index, char in enumerate(unique_char)} # empty spaces are 0 as the most are empty spaces.

    seq_int = [char_to_index[w] for w in file1] # convert a book into a sequence of integers for processing

    # convert seq_int to interpretable sequences
    language_sequence = np.array(seq_int).reshape([len(seq_int),1,1])

    print(language_sequence)
    cg = Chunking_Graph(DT=0.00001, theta=0.999996)  # initialize chunking part with specified parameter
    DATA = {}
    DATA['N'] = []
    DATA['chunk learned'] = []
    # show that the words learned are different between different stages of learning
    intr = 1000
    for i in range(200,400):
        # interval of 50000 words
        cg = learn_stc_classes(language_sequence[intr*i:intr*(i+1),:,:],cg)
        print('after learning this number of iterations ', intr*i)
        chunks = []
        for chunk in list(cg.M.keys()):
            ckarr = tuple_to_arr(chunk)
            backwardindex = list(np.ravel(ckarr))
            seq_word = [index_to_char[i] for i in backwardindex]
            print('learned chunk: ', "".join(seq_word))
            chunks.append("".join(seq_word))
        DATA['chunk learned'].append(chunks)
        DATA['N'].append(intr*(i+1))
        print( 'start next round')
    return


text_rep_learning()