# demonstrate the acquisition of previous noun category will help children learn novel artificial nouns
import numpy as np
from collections import Counter
from Generative_Model import *
from Learning import *
from CG1 import *
from chunks import *

new_noun = {'wuggy': 1,
            'toma': 2,
            'peri': 3,
            'gazzer': 4,}

old_noun = {'cookie':5,
            'bird': 6,
            'bug': 7,
            'cat': 8}

others = {'kissing': 9,
        'pushing': 10,
        'washing': 11,
        'brushing': 12,
          'the': 13,
          's':14,
          '.':15,
          'is ': 16}

def translate_recalled_seq(recalled_seq,decoder):
    decoded_seq = []
    for i in range(0, recalled_seq.shape[0]):
        decoded_seq.append(decoder[recalled_seq[i,0,0]])
    return decoded_seq



training_seq = []
n_iter = 100
for i in range(0, n_iter):
    s1_train = ['the'] + [str(np.random.choice(list(old_noun.keys())))] + ['.']
    s2_train = ['the'] + [np.random.choice(list(old_noun.keys()))] + ['is '] + [
        np.random.choice(['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [
                   np.random.choice(list(old_noun.keys()))] + ['.']
    s3_train = [np.random.choice(list(old_noun.keys()))] + ['s']

    training_seq += [s1_train, s2_train, s3_train][np.random.choice([0,1,2])]

print(training_seq)
# calculate before testing: 1. with definitive article: wuggy, toma, peri, gazzer
# being the subject, being the object
# adding s

n_iter = 30

# the agent role
test_seq = []
for i in range(0, n_iter):
    s1_test = ['the'] + [str(np.random.choice(list(new_noun.keys())))] + ['is '] + [np.random.choice(
        ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(old_noun.keys()))] + ['.']
    test_seq += s1_test

# # the agent role
# test_seq_agent = []
# for i in range(0, n_iter):
#     s1_test = ['the'] + [str(np.random.choice(list(new_noun.keys())))] + ['is '] + [np.random.choice(
#         ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(old_noun.keys()))] + ['.']
#     test_seq_agent += s1_test
#
# # patient role
# test_seq_patient = [] # the new noun plays the agent role
# for i in range(0, n_iter):
#     s1_test = ['the'] + [str(np.random.choice(list(old_noun.keys())))] + ['is '] + [np.random.choice(
#         ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(new_noun.keys()))] + ['.']
#     test_seq_patient += s1_test
#
# # plural role
# test_seq_patient = [] # the new noun plays the agent role
# for i in range(0, n_iter):
#     s1_test = ['the'] + [str(np.random.choice(list(old_noun.keys())))] + ['is '] + [np.random.choice(
#         ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(new_noun.keys()))] + ['.']
#     test_seq_patient += s1_test

# translate the training and testing sequence into tokens
vocab = {**new_noun, **old_noun, **others}

decoder = {value: key for key, value in vocab.items()}
full_seq_train = [vocab[key] for key in training_seq]
full_seq_test = [vocab[key] for key in test_seq]
fullseq = np.array(full_seq_train).reshape([len(full_seq_train), 1, 1])


n_iter = 10


# learn sequences with the chunking graph
i = 0
cg = CG1(DT=0.1, theta=0.96)

cg, chunkrecord = hcm_learning(fullseq, cg,
                               abstraction = True)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

recalled_seq, ps = recall(cg, seql=20, firstitem=fullseq[0, 0, 0])
# transfer sequences with the chunking graph
# need to concatinate train and test

print(translate_recalled_seq(recalled_seq, decoder))

fullseq = np.array(full_seq_test).reshape([len(full_seq_test), 1, 1])

cg, chunkrecord = hcm_learning(fullseq, cg,
                               abstraction = True)  # with the rational chunk models, rational_chunk_all_info(seq, cg)


recalled_seq, ps = recall(cg, seql=100, firstitem=fullseq[0, 0, 0])


# implement filters to check the frequencies of words at different text positions
recalled_seq = translate_recalled_seq(recalled_seq, decoder)

print(recalled_seq)

'''category: a dictionary of words in a particular grammatical category'''
new_nouns = list(new_noun.keys())
verbs = ['kissing', 'pushing', 'washing', 'brushing']
Agent = {key: 0 for key in list(new_noun.keys())}
Patient, PluralMorph, ProdArgu = Agent.copy(), Agent.copy(), Agent.copy()
for i in range(0, len(recalled_seq) - 5):
    if recalled_seq[i] == 'the' and recalled_seq[i + 1] in new_nouns and recalled_seq[i + 2] == 'is ' and recalled_seq[
        i + 3] in verbs and recalled_seq[i + 4] == 'the':
        Agent[recalled_seq[i + 1]] += 1
    if recalled_seq[i] == 'the' and recalled_seq[i + 2] == 'is ' and recalled_seq[i + 3] in verbs and recalled_seq[i + 4] == 'the' and recalled_seq[i + 5] in new_nouns:
        Patient[recalled_seq[i + 5]] += 1
    if recalled_seq[i] == 'the' and recalled_seq[i + 1] in new_nouns and recalled_seq[i+2] == '.':
        ProdArgu[recalled_seq[i + 1]] += 1
    if recalled_seq[i] in new_nouns and recalled_seq[i + 1] == 's':
        PluralMorph[recalled_seq[i + 1]] += 1


Test_type = ['Agent', 'Patient', 'Definitive Article', 'Plural Morphology']

# essentially, evaluate the frequencies of how often each utterance type is used in the generative sequence produced by the model

import matplotlib.pyplot as plt
import numpy as np

# Sample data
new_words = list(new_noun.keys())
values_agent = list(Agent.values())
values_patient = list(Patient.values())
values_prod_argu = list(ProdArgu.values())
values_plural_morp = list(PluralMorph.values())

# Number of categories
n = len(new_words)

# The x locations for the groups
x = np.arange(n)

# The width of the bars
width = 0.2

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each set of bars
bars1 = ax.bar(x - 1.5 * width, values1, width, label='S', color='skyblue')
bars2 = ax.bar(x - 0.5 * width, values2, width, label='Series 2', color='salmon')
bars3 = ax.bar(x + 0.5 * width, values3, width, label='Series 3', color='lightgreen')
bars4 = ax.bar(x + 1.5 * width, values4, width, label='Series 4', color='gold')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Category')
ax.set_ylabel('Frequency')
ax.set_title('Novel Noun Application')
ax.set_xticks(x)
ax.set_xticklabels(new_words)
ax.legend()

# Show the plot
plt.show()






hist_matrix = np.zeros(shape=(4,4))
for neuronname in quadruplets:
    firing_freq_ff = sum(enlarged_population_spike_rate_df.loc[neuronname]['spike_rate']* event_tag['ff'])/100
    firing_freq_nf = sum(enlarged_population_spike_rate_df.loc[neuronname]['spike_rate']* event_tag['nf'])/100
    firing_freq_fn = sum(enlarged_population_spike_rate_df.loc[neuronname]['spike_rate']* event_tag['fn'])/100
    firing_freq_nn = sum(enlarged_population_spike_rate_df.loc[neuronname]['spike_rate']* event_tag['nn'])/100

    if firing_freq_nn>0: hist_matrix[0,0] = hist_matrix[0,0] + 1
    else: non_responding_matrix[0,0] = non_responding_matrix[0,0] + 1
    if firing_freq_nf>0: hist_matrix[0,1] = hist_matrix[0,1] + 1
    else: non_responding_matrix[0,1] = non_responding_matrix[0,1] + 1
    if firing_freq_fn>0: hist_matrix[1,0] = hist_matrix[1,0] + 1
    else: non_responding_matrix[1,0] = non_responding_matrix[1,0] + 1
    if firing_freq_ff>0: hist_matrix[1,1] = hist_matrix[1,1] + 1
    else: non_responding_matrix[1,1] = non_responding_matrix[1,1] + 1

# Step 4: Convert to DataFrame
hist_freq = pd.DataFrame(hist_matrix, columns=['Agent', 'Patient','Definitive Article','Plural Morphology'], index = ['Agent', 'Patient','Definitive Article','Plural Morphology'])

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(hist_freq, cmap='coolwarm')
plt.title('Number of Responding Quadruplet Ensembles')
plt.show()

print(training_seq)
