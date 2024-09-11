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
n_iter = 200
for i in range(0, n_iter):
    s1_train = ['the'] + [str(np.random.choice(list(old_noun.keys())))] + ['.']
    s2_train = ['the'] + [np.random.choice(list(old_noun.keys()))] + ['is '] + [
        np.random.choice(['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [
                   np.random.choice(list(old_noun.keys()))] + ['.']
    s3_train = [np.random.choice(list(old_noun.keys()))] + ['s']
    training_seq += [s1_train, s2_train, s3_train][np.random.choice([0,1,2])]

# calculate before testing: 1. with definitive article: wuggy, toma, peri, gazzer
# being the subject, being the object
# adding s
# original/overall
# test_seq = []
# for i in range(0, n_iter):
#     s1_test = ['the'] + [str(np.random.choice(list(new_noun.keys())))] + ['is '] + [np.random.choice(
#         ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(old_noun.keys()))] + ['.']
#     test_seq += s1_test

n_iter = 10
# the agent role
test_seq_agent = []
for i in range(0, n_iter):
    s1_test = ['the'] + [str(np.random.choice(list(new_noun.keys())))] + ['is '] + [np.random.choice(
        ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(old_noun.keys()))] + ['.']
    test_seq_agent += s1_test

# patient role
test_seq_patient = [] # the new noun plays the agent role
for i in range(0, n_iter):
    s1_test = ['the'] + [str(np.random.choice(list(old_noun.keys())))] + ['is '] + [np.random.choice(
        ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(new_noun.keys()))] + ['.']
    test_seq_patient += s1_test

# plural role
test_seq_plural = [] # the new noun plays the agent role
for i in range(0, n_iter):
    s1_test = [np.random.choice(list(new_noun.keys()))] + ['s']
    test_seq_plural += s1_test

# definitive article
test_seq_def = [] # the new noun plays the agent role
for i in range(0, n_iter):
    s1_test = ['the'] + [np.random.choice(list(new_noun.keys()))] + ['.']
    test_seq_def += s1_test

# translate the training and testing sequence into tokens
vocab = {**new_noun, **old_noun, **others}

decoder = {value: key for key, value in vocab.items()}

test_type = ['Agent', 'Patient', 'Definitive Article', 'Plural Morphology']
test_pipeline = {'Agent': test_seq_agent, 'Patient': test_seq_patient, 'Definitive Article': test_seq_def, 'Plural Morphology': test_seq_plural}
hist_matrix = np.zeros(shape=(4, 4))
for j in range(0,4):
    t = test_type[j]
    ############# Training #############
    full_seq_train = [vocab[key] for key in training_seq]
    fullseq = np.array(full_seq_train).reshape([len(full_seq_train), 1, 1])  # learn sequences with the chunking graph
    i = 0
    cg = CG1(DT=0.1, theta=0.96)
    cg, chunkrecord = hcm_learning(fullseq, cg,
                                   abstraction=True)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    recalled_seq, ps = recall(cg, seql=20, firstitem=fullseq[0, 0, 0])
    print(translate_recalled_seq(recalled_seq, decoder))
    ############# Test #################
    test_seq = test_pipeline[t]
    full_seq_test = [vocab[key] for key in test_seq]
    fullseq = np.array(full_seq_test).reshape([len(full_seq_test), 1, 1])
    cg, chunkrecord = hcm_learning(fullseq, cg, abstraction=True)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
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
        if recalled_seq[i] == 'the' and recalled_seq[i + 1] in new_nouns and recalled_seq[i + 2] == 'is ' and recalled_seq[i + 3] in verbs and recalled_seq[i + 4] == 'the':
            Agent[recalled_seq[i + 1]] += 1
        if recalled_seq[i] == 'the' and recalled_seq[i + 2] == 'is ' and recalled_seq[i + 3] in verbs and recalled_seq[i + 4] == 'the' and recalled_seq[i + 5] in new_nouns:
            Patient[recalled_seq[i + 5]] += 1
        if recalled_seq[i] == 'the' and recalled_seq[i + 1] in new_nouns and recalled_seq[i+2] == '.':
            ProdArgu[recalled_seq[i + 1]] += 1
        if recalled_seq[i] in new_nouns and recalled_seq[i + 1] == 's':
            PluralMorph[recalled_seq[i]] += 1
    hist_matrix[j, 0] = sum(list(Agent.values()))
    hist_matrix[j, 1] = sum(list(Patient.values()))
    hist_matrix[j, 2] = sum(list(ProdArgu.values()))
    hist_matrix[j, 3] = sum(list(PluralMorph.values()))


# Step 4: Convert to DataFrame
hist_freq = pd.DataFrame(hist_matrix, columns=['Agent', 'Patient','Definitive Article','Plural Morphology'], index = ['Agent', 'Patient','Definitive Article','Plural Morphology'])

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(hist_freq, cmap='coolwarm')
plt.title('Number of ')
plt.show()

print(training_seq)
