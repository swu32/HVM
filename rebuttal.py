import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from collections import Counter


# TODO: sensitivity analysis
# TODO:

# simulate generative sequence and learning curves when the sequence length is 5000, be careful not to overwrite the older datasets

# depth = 30, seql=5000, alphabet = 10
# datahcm './data/hcm_fixed_support_set' + ' d = ' + str(d) + ' seql=5000.npy'
# datahvm './data/hvm_fixed_support_set' + ' d = ' + str(d) + ' seql=5000.npy'
# datalz_encoding_bits './data/lz_encoding_bits' + ' d = ' + str(d) + ' seql=5000.npy'
# Ground truth './data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(10) + ' seql=5000.npy'




# Comparison with HPYP:
# generate text document from the existing generative model
# 


def read_wikitext_file(filepath):
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read()
    return text



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



def sequence_complexity_comparison_longer_sequence_length():
    # bar plot comparison of sequence complexity comparison between different encoding algorithms
    d = 30
    sequence_length = 1000
    datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '_seql=' + str(sequence_length) + '.npy')
    datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '_seql=' + str(sequence_length) + '.npy')
    datalz_encoding_bits = np.load('./data/lz_encoding_bits_d=30'+'_seql=' + str(sequence_length) + '.npy')
    data_raw_encoding_bits = np.load('./data/sequence_original_encoding_bits_d=30seql=5000.npy')
    GT = np.load('./data/generative_hvm' + ' d = ' + str(30) + 'sz = ' + str(10) + '.npy')
    seql = sequence_length
    bit_per_symbol = 12

    hcm_mean_seq_l = seql / np.mean(datahcm[:, -1, 3], axis=0)  # at the end of training
    hvm_mean_seq_l = seql / np.mean(datahvm[:, -1, 3], axis=0)   # average over different runs
    lz_mean_seq_l = np.mean(datalz_encoding_bits[:, -1, 0], axis=0)/bit_per_symbol  # average over different runs
    gt_seq_l = seql / GT[0, 3]  # the number of symbols to encode sequences in ground truth
    raw_mean_seq_l = np.mean(data_raw_encoding_bits[:, -1, 0], axis=0)/bit_per_symbol#
    sem_seq_l = [calculate_sem(seql/datahcm[:, -1, 3]), calculate_sem(seql/datahvm[:, -1, 3]),
                      calculate_sem(datalz_encoding_bits[:, -1, 0]/bit_per_symbol), 0]

    datalz_complexity = np.load('./data/lz_complexity_d=30'+'_seql=' + str(sequence_length) + '.npy')
    GT = np.load('./data/generative_hvm' + ' d = ' + str(30) + 'sz = ' + str(10) + '.npy')
    data_gt_complexity = [GT[0, 4]]

    hcm_mean_complexity = np.mean(datahcm[:, -1, 4], axis=0)  # average over different runs
    hvm_mean_complexity = np.mean(datahvm[:, -1, 4], axis=0)  # average over different runs
    lz_mean_complexity = np.mean(datalz_complexity[:, -1, 0], axis=0)  # average over different runs
    gt_mean_complexity = data_gt_complexity[0]
    sem_complexity = [calculate_sem(datahcm[:, -1, 4]), calculate_sem(datahvm[:, -1, 4]), calculate_sem(datalz_complexity[:, -1, 0]), 0]

    data_gt_entropy = [GT[0, 5]]
    hcm_mean_entropy = np.mean(datahcm[:, -1, 5], axis=0)
    hvm_mean_entropy = np.mean(datahvm[:, -1, 5], axis=0)
    gt_mean_entropy = data_gt_entropy[0]
    sem_seq_entropy = [calculate_sem(datahcm[:, -1, 5]), calculate_sem(datahvm[:, -1, 5]), 0]

    plt.rcParams['font.size'] = 18
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 3 rows, 1 column
    # Adjust subplot parameters
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase wspace for more horizontal space
    models = ['Without \n Abstraction', 'With \n Abstraction']
    parsing_length = [np.mean(datahvm[:, -1, 6], axis=0), np.mean(datahvm[:, -1, 1], axis=0)]
    sem_parsing_length = [calculate_sem(datahvm[:, -1, 6]), calculate_sem(datahvm[:, -1, 1])]
    axes[0].bar(models, parsing_length, color=['skyblue','royalblue'], edgecolor = 'black', yerr = sem_parsing_length)
    axes[0].set_ylabel('Parsing Search Steps')  # Bigger font size for y-axis label
    axes[0].set_yscale('log')

    # Sequence Length
    models = ['HCM', 'HVM', 'LZ78', 'GT']
    seq_length = [hcm_mean_seq_l, hvm_mean_seq_l, lz_mean_seq_l, gt_seq_l]
    axes[1].bar(models, seq_length, color=['#CC5500', 'royalblue', '#36454F','forestgreen'], edgecolor = 'black', yerr = sem_seq_l)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Parsing Length |W|')
    plt.yscale('linear')

    # Sequence Complexity
    models = ['HCM', 'HVM', 'LZ78', 'GT']
    seq_complexity = [hcm_mean_complexity, hvm_mean_complexity, lz_mean_complexity, gt_mean_complexity]
    axes[2].bar(models, seq_complexity, color=['#CC5500', 'royalblue', '#36454F', 'forestgreen'], edgecolor = 'black', yerr = sem_complexity)
    axes[2].set_xlabel('Models')
    axes[2].set_ylabel('Sequence Likelihood -logP(S)')
    plt.yscale('linear')

    # Coding efficiency
    models = ['HCM', 'HVM', 'LZ78', 'GT']
    seq_complexity = [hcm_mean_complexity, hvm_mean_complexity, lz_mean_complexity, gt_mean_complexity]
    axes[3].bar(models, seq_complexity, color=['#CC5500', 'royalblue', '#36454F', 'forestgreen'], edgecolor = 'black', yerr = sem_complexity)
    axes[3].set_xlabel('Models')
    axes[3].set_ylabel('')
    plt.yscale('linear')

    # Sequence Complexity
    models = ['HCM', 'HVM', 'GT']
    seq_complexity = [hcm_mean_entropy, hvm_mean_entropy, gt_mean_entropy]
    plt.figure()
    plt.bar(models, seq_complexity, color='royalblue', edgecolor='black', yerr= sem_seq_entropy)
    plt.xlabel('Models')
    plt.ylabel('Parsing Entropy')
    plt.show()
    return


# load generative data
openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = 10.npy'
with open(openpath, 'rb') as f:
    fullseq = np.load(f)

seql = 1000
# Flatten the array to 1D
flattened_array = fullseq[:seql,:,:].flatten()
# Your initial string of integers
string_of_integers = ' '.join(map(str, flattened_array.astype(int)))

# Split the string into a list of integers
integers = list(map(int, string_of_integers.split()))

# Map each integer to a letter using ASCII values
# Assuming integers are in the range of 0-25, which correspond to 'A'-'Z'
string_of_letters = ' '.join(chr(i + 65) for i in integers)

print(string_of_letters)


# Save the string to a text file
with open("./hpylm-python/src/generative_sequence.txt", "w") as file:
    file.write(string_of_letters)
# store as text document
bnc = 'bnc_spoken.train'
gutenberg = 'gutenberg.train'
childes = 'childes.train'
opensubtitles = 'open_subtitles.train'

for name in [bnc, gutenberg, childes, opensubtitles]:
    train_path = '/Users/swu/Documents/MouseHCM/train_10M/' + name
    train_text = read_wikitext_file(train_path)
    with open("./hpylm-python/src/shortened"+name+".txt", "w") as file:
        file.write(' '.join(train_text[:1000]))

sequence_complexity_comparison_longer_sequence_length()
