from collections import Counter
from Learning import *
from CG1 import *
import numpy as np
import os
import json


def store_subj_data(cghvm, subj):
    decoder = {value: key for key, value in vocab.items()}

    file_chunks = "./data/fMRI_data/fMRI_chunk_text subj = " + str(subj) + '.json'
    file_variables = "./data/fMRI_data/fMRI_variable_text subj = " + str(subj) + '.json'

    chunkdata = []  # ([chunk content], count)
    # store learned chunks
    for chunkname, chunk in cghvm.chunks.items():
        chunkcontent = []
        for content in chunk.ordered_content:
            if isinstance(content, set):
                for contenttuple in sorted(list(content)):
                    contenttoprint = translate_chunks_to_words(contenttuple, decoder)
                    chunkcontent.append(contenttoprint)
            else:  # a variable, content type os a string
                chunkcontent.append(content)
        chunkdata.append([chunkcontent, chunk.count])

    # Write the list of lists to a file
    with open(file_chunks, 'w') as file:
        json.dump(chunkdata, file)

    # Read the list of lists back from the file
    with open(file_chunks, 'r') as file:
        loaded_list_of_lists = json.load(file)

    vardata = []  # (varname, [[entailingchunk, count],... ])
    # store learned variables
    for varname, var in cghvm.variables.items():
        var_property = []
        var_property.append(varname)
        entailinginfo = []
        for chunkkey, chunk in var.entailingchunks.items():
            chunkcontent = []
            for content in chunk.ordered_content:
                if isinstance(content, set):
                    for contenttuple in sorted(list(content)):
                        contenttoprint = translate_chunks_to_words(contenttuple, decoder)
                        chunkcontent.append(contenttoprint)
                else:  # a variable, content type os a string
                    chunkcontent.append(content)
            entailinginfo.append([chunkcontent, var.chunk_probabilities[chunk]])
        var_property.append(entailinginfo)

        vardata.append(var_property)
    # Write the list of lists to a file
    with open(file_variables, 'w') as file:
        json.dump(vardata, file)
    return


def plot_model_comparison(datahcm, datahvm, savename = 'modelcomparisonfMRI.png'):
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
        hcm_mean = np.mean(datahcm[:, :, i + 1], axis = 0)
        hvm_mean = np.mean(datahvm[:, :, i + 1], axis = 0)
        ax.plot(x, hcm_mean, label='HCM', color='orange', linewidth=4, alpha=0.3)
        ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha=0.3)
        for j in range(0, datahcm.shape[0]):
            ax.plot(x, datahcm[j, :, i + 1], color='orange', linewidth=1, alpha = 0.3)
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

# Specify the path to the file
file_path = "../maximal_activation_patterns.txt"


# Open the file for reading
with open(file_path, 'r') as file:
    # Read all lines into a list, stripping the newline characters
    brain_activities = [line.strip() for line in file]

print("The list of strings read from the file:", brain_activities)


def build_vocabulary(tokens, min_freq=0):
    # Count the occurrence of each word in the dataset
    word_counts = Counter(tokens)
    vocabulary = {word: i for i, (word, freq) in enumerate(word_counts.items()) if freq >= min_freq}
    return vocabulary

def translate_chunks_to_words(contenttuple, decoder):
    return decoder[contenttuple[3]]



# Build the vocabulary
vocab = build_vocabulary(brain_activities)
train_tokens = [vocab[w] for w in brain_activities]
wholeseq = np.array(train_tokens).reshape([-1,1,1])
# Print some information about the dataset and vocabulary

n_measure = 9
n_run = 1  # int(len(fullseq)/slice_sz)
n_iter = 8
n_sub = 10
datahcm = np.empty((n_sub, n_iter, n_measure))
datahvm = np.empty((n_sub, n_iter, n_measure))
for subj in [0]:#range(0, 155):
    R = 10
    seq = wholeseq[168*subj:168*(subj+R), :, :]
    cghvm = CG1(DT=0.1, theta=0.996)
    cghvm = hcm_markov_control_1(seq, cghvm, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    datahvm[subj, :, :] = np.array(cghvm.learning_data)
    datahcm[subj, :, :] = np.array(cghcm.learning_data)

    arrseq = np.array(wholeseq).reshape([-1,1,1])[168 * subj:168 * (subj + R), :, :]
    seq, seql = convert_sequence(arrseq[:, :, :])  # loaded with the 0th observation
    cghvm.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
    cghvm, chunkrecord = parse_sequence(cghvm, seq, arrseq, seql=seql)
    cghvm.rep_cleaning()
    store_subj_data(cghvm, subj)

# save learning data
np.save('./data/hvm_fMRI.npy', datahvm)
np.save('./data/hcm_fMRI.npy', datahcm)
plot_model_comparison(datahcm, datahvm, savename='./data/fMRI_data/fMRI_learning_curve.png') # like, in this case, this is the averge learning curve?






def printchunk(chunk, text_file, level=0):
    index = 0
    for content in chunk.ordered_content:
        if isinstance(content, set):
            for contenttuple in sorted(list(content)):
                contenttoprint = translate_chunks_to_words(contenttuple, decoder)
                print("\t" * level + str(index) + ": " + contenttoprint)
                text_file.write("\t" * level + str(index) + ": " + contenttoprint + '\n')

        else:  # a variable, content type os a string
            for chunk in list(cghvm.variables[content].entailingchunks.values()):
                if cghvm.variables[content].chunk_probabilities[chunk]>0:
                    print("\t" * (level+1) + str(index) + ": " + '------------------')
                    text_file.write("\t" * (level+1) + str(index) + ": " + '------------------'  + '\n')
                    printchunk(chunk, text_file, level=level + 1)
        index = index + 1
    return

# print out learned chunks
decoder = {value: key for key, value in vocab.items()}
text_file = open("fMRI_abstraction_learning.txt", "w")



min_len = 1
min_count = 10
for chunk in list(cghvm.chunks.values()):
    if chunk.T > min_len and chunk.count >= min_count:
        print('****************'+' freq = ' + str(chunk.count))
        text_file.write('****************' + ' freq = ' + str(chunk.count) + '\n')
        printchunk(chunk, text_file, level=0)
text_file.close()



# rank the chunks by parsing count
# for each chunk, output chunkpart + variable part,
# the name of the chunks that each variable entails,
print()