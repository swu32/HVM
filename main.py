from Generative_Model import *
from text_learning import *
from Learning import *
from CG1 import *
from chunks import *
import numpy as np
import os
from time import time
from chunks import *


def evaluate_perplexity(data, chunkrecord):
    # TODO: convert chunkrecord into sequence of probability
    p = []
    n_ck = 0
    for t in range(0, len(data)):
        if t in list(chunkrecord.keys()):
            freq = chunkrecord[t][0][1]
            n_ck = n_ck + 1
            p.append(freq / n_ck)
        else:  # a within-chunk element
            p.append(1)
    perplexity = 2 ** (-np.sum(np.log2(np.array(p))) / len(p))

    return perplexity


def plot_model_learning_comparison(cg1, cg2):
    import matplotlib.pyplot as plt
    import numpy as np

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables', 'storage cost']
    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
    ld1 = np.array(cg1.learning_data)
    ld2 = np.array(cg2.learning_data)
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(ld1[:, 0])

    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        y1 = ld1[:, i + 1]
        y2 = ld2[:, i + 1]
        ax.plot(x, y1, label='HCM')
        ax.plot(x, y2, label='HVM')
        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig('modelcomparison.png')
    return


def test_random_graph_abstraction(generation=False, sequence_length=1000, depth_increment=[5, 10, 15, 20, 25, 30, 35]):
    if generation:
        for d in depth_increment:
            random_abstract_representation_graph(save=True, alphabet_size=10, depth=d, seql=sequence_length)

    # with open('random_abstract_sequence.npy', 'rb') as f:
    #     seq = np.load(f)
    for sz in depth_increment:

        # openpath = './generative_sequences_different_parameters/random_abstract_sequence_a='+ str(10) + '_d='+ str(sz) + '_p_variable=' + str(0.5) +'_seql=' + str(sequence_length) + '.npy'
        openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = sequence_length
        n_measure = 9
        n_run = 5  # int(len(fullseq)/slice_sz)
        n_iter = 20
        datahcm = np.empty((n_run, n_iter, n_measure))
        datahvm = np.empty((n_run, n_iter, n_measure))
        i = 0
        for seq in slicer(fullseq, slice_sz):
            if i == n_run: break  # just do 1 iteration
            cghvm = CG1(DT=0.1, theta=0.996)
            cghvm = hcm_markov_control_1(seq, cghvm,
                                         MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            datahvm[i, :, :] = np.array(cghvm.learning_data)

            cghcm = CG1(DT=0.1, theta=0.996)
            cghcm = hcm_markov_control_1(seq, cghcm, ABS=False,
                                         MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            datahcm[i, :, :] = np.array(cghcm.learning_data)

            i = i + 1
        np.save('./data/hcm_fixed_support_set' + ' d = ' + str(sz) + '_seql=' + str(sequence_length) + '.npy', datahcm)
        np.save('./data/hvm_fixed_support_set' + ' d = ' + str(sz) + '_seql=' + str(sequence_length) + '.npy', datahvm)
        plot_average_model_learning_comparison(datahcm, datahvm, d=sz,
                                               savename='./data/fixed_support_set' + ' d = ' + str(sz) + '.png',
                                               gt_load_name='./data/generative_hvm_a=10_d=' + str(
                                                   sz) + '_p_variable=' + str(
                                                   0.5) + '_seql=' + str(1000) + '.npy')
    return


def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main():
    test_random_graph_abstraction(generation=False, sequence_length=1000, depth_increment=[30])

    test_random_graph_abstraction(generation=False)
    seq = abstraction_illustration()

    test_depth_parsing()
    simonsaysex2()
    test_simple_abstraction()  # within which there is an hcm rational

    pass


if __name__ == "__main__":
    main()
