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
test_seq = []
for i in range(0, n_iter):
    s1_test = ['the'] + [str(np.random.choice(list(new_noun.keys())))] + ['is '] + [np.random.choice(
        ['kissing', 'pushing', 'washing', 'brushing'])] + ['the'] + [np.random.choice(list(new_noun.keys()))] + ['.']
    test_seq += s1_test



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
print(translate_recalled_seq(recalled_seq, decoder))




print(training_seq)