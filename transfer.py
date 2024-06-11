from Hand_made_generative import *
from Generative_Model import *
from text_learning import *
from Learning import *
from CG1 import *
from chunks import *
import numpy as np
import PIL as PIL
from PIL import Image
import os
from time import time
from chunks import *
from abstraction_test import *
import matplotlib.pyplot as plt

def measure_KL():
    '''Measurement of kl divergence across learning progress
    n_sample: number of samples used for a particular uncommital generative model
    d: depth of the generative model
    n: length of the sequence used to train the learning model'''
    df = {}
    df['N'] = []
    df['kl'] = []
    df['type'] = []
    df['d'] = []
    n_sample = 1  # eventually, take 100 runs to show such plots
    n_atomic = 5
    ds = [3, 4, 5, 6, 7, 8]
    Ns = np.arange(100,3000,100)
    for d in ds: # varying depth, and the corresponding generative model it makes
        depth = d
        for i in range(0, n_sample):
            # in every new sample, a generative model is proposed.
            cg_gt = generative_model_random_combination(D=depth, n=n_atomic)
            cg_gt = to_chunking_graph(cg_gt)
            for n in Ns:
                print({' d ': d, ' i ': i, ' n ': n })
                # cg_gt = hierarchy1d() #one dimensional chunks
                seq = generate_hierarchical_sequence(cg_gt.M, s_length=n)
                cg = Chunking_Graph(DT=0, theta=1)  # initialize chunking part with specified parameters
                cg = rational_chunking_all_info(seq, cg)
                imagined_seq = cg.imagination(n, sequential=True, spatial=False, spatial_temporal=False)
                kl = evaluate_KL_compared_to_ground_truth(imagined_seq, cg_gt.M, Chunking_Graph(DT=0, theta=1))

                # take in data:
                df['N'].append(n)
                df['d'].append(depth)
                df['kl'].append(kl)
                df['type'].append('ck')

    df = pd.DataFrame.from_dict(df)
    df.to_pickle('KL_rational_learning_N')  # where to save it, usually as a .pkl
    return df



def plot_model_learning_comparison(cg1, cg2):
    import matplotlib.pyplot as plt
    import numpy as np

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables','storage cost']
    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable','bits']
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



def test_random_graph_abstraction():
    d = 10
    openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(d) + '.npy'
    with open(openpath, 'rb') as f:
        fullseq = np.load(f)
    slice_sz = 1000
    n_measure = 9
    n_run = 1#int(len(fullseq)/slice_sz)
    n_iter = 10
    datahcm = np.empty((n_run, n_iter, n_measure))
    datahvm = np.empty((n_run, n_iter, n_measure))
    i = 0
    seq = slicer(fullseq, slice_sz)[0] # try on one sequence first

    for it in range(0, n_iter):
        cghvm = CG1(DT=0.1, theta=0.996)
        cghvm = hcm_markov_control_1(seq, cghvm, MAXit=it)
        cghcm = CG1(DT=0.1, theta=0.996)
        cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit=it)


    for seq in slicer(fullseq, slice_sz):
        if i == n_run: break # just do 1 iteration
        cghvm = CG1(DT=0.1, theta=0.996)
        cghvm = hcm_markov_control_1(seq, cghvm, MAXit = n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

        cghcm = CG1(DT=0.1, theta=0.996)
        cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit = n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)


        # datahcm[i,:,:] = np.array(cghcm.learning_data)
        # datahvm[i,:,:] = np.array(cghvm.learning_data)
        i = i + 1
    # np.save('./data/hcm_fixed_support_set' + ' d = ' + str(sz) + '.npy', datahcm)
    # np.save('./data/hvm_fixed_support_set' + ' d = ' + str(sz) + '.npy', datahvm)
    plot_average_model_learning_comparison(datahcm, datahvm, d = sz, savename = './data/fixed_support_set' + ' d = ' + str(sz) + '.png')
    return



def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def relative_likelihood_existing_sequence():
    # compute abstraction advantage purely based on existing data.

    d  = 10
    datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    x = np.arange(0, datahcm.shape[1],1)

    hcm_mean_complexity = np.mean(datahcm[:, :, 4], axis=0)  # average over different runs
    hvm_mean_complexity = np.mean(datahvm[:, :, 4], axis=0)  # average over different runs
    # calculate standard error
    hcm_se_complexity = stats.sem(datahcm[:, :, 4], axis=0)
    hvm_se_complexity = stats.sem(datahvm[:, :, 4], axis=0)  # average over different runs

    plt.errorbar(x, hcm_mean_complexity, yerr=hcm_se_complexity, label='HCM', color='orange', linewidth=3, fmt='-o')
    plt.errorbar(x, hvm_mean_complexity, yerr=hvm_se_complexity, label='HVM', color='blue', linewidth=3, fmt='-o')
    plt.xlabel('Level of Abstraction')
    plt.ylabel('Transfer Sequence Likelihood')
    # Show the figure
    plt.legend(loc='best')
    plt.show()




    plt.figure()
    relative_complexity = np.mean(datahcm[:, :, 4] - datahvm[:, :, 4], axis=0)  # average over different runs
    relative_se_complexity = stats.sem(datahcm[:, :, 4] - datahvm[:, :, 4], axis=0)

    plt.errorbar(x, relative_complexity, yerr=relative_se_complexity, color='purple', linewidth=3, fmt='-o')
    plt.xlabel('Level of Abstraction')
    plt.ylabel('Transfer Sequence Likelihood Relative Advantage')
    # Show the figure
    plt.legend(loc='best')
    plt.show()
    return


def evaluate_transfer(arayseq, cg):
    """evaluate the transfer performance of cg on a transfer arayseq (the last list in learning data is transfer performance)"""
    seql, H, W = arayseq.shape
    cg.update_hw(H, W) #update initial parameters
    seq, seql = convert_sequence(arayseq[:, :, :])# loaded with the 0th observation
    cg.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
    cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
    cg = evaluate_representation(cg, chunkrecord, seql)  # evaluate only during chunking iterations
    return cg


def comparisonplot(datahcm, datahvm, total_n_iter):
    # Sample data: means and standard deviations for two algorithms across three datasets
    labels = np.arange(0, total_n_iter, 1)
    means_hcm = datahcm[0, :, 4]
    std_hcm = [0] * len(labels)  # surrogate for now
    means_hvm = datahvm[0, :, 4]
    std_hvm = [0] * len(labels)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, means_hcm, width, label='HCM', color='Orange', yerr=std_hcm, capsize=5)
    rects2 = ax.bar(x + width / 2, means_hvm, width, label='HVM', color='Blue', yerr=std_hvm, capsize=5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Layers of Abstraction')
    ax.set_ylabel('Negative Log Likelihood on Transfer Sequences')
    ax.set_title('Transfer Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

    return



def transfer_different_sequence_same_generative_model():
    d = 10
    openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(d) + '.npy'
    with open(openpath, 'rb') as f:
        fullseq = np.load(f)

    slice_sz = 1000
    n_measure = 9
    n_run = 20  # int(len(fullseq)/slice_sz)
    total_n_iter = 10

    datahcm = np.empty((n_run, total_n_iter, n_measure))  # transfer performance on the novel sequences (from the same generative model)
    datahvm = np.empty((n_run, total_n_iter, n_measure))  # after training the model on n iteratons with or without abstraction

    for i in range(0, n_run):
        for nit in range(total_n_iter):
            n_iter = nit# iteration to learn abstraction
            # training models
            seq = fullseq[0:0 + slice_sz]

            cghvm = CG1(DT=0.1, theta=0.996)
            cghvm = hcm_markov_control_1(seq, cghvm, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            cghcm = CG1(DT=0.1, theta=0.996)
            cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            # testing seq, different sequence from the same generative model, evaluate both models on the testing sequence
            seq =  fullseq[1000:1000 + slice_sz] # evaluation on the same sequence
            cghcm = evaluate_transfer(seq, cghcm)
            cghvm = evaluate_transfer(seq, cghvm)

            datahcm[i, nit, :] = np.array(cghcm.learning_data[-1])
            datahvm[i, nit, :] = np.array(cghvm.learning_data[-1])
    comparisonplot(datahcm, datahvm, total_n_iter)
    return
transfer_different_sequence_same_generative_model()

relative_likelihood_existing_sequence()
test_random_graph_abstraction(generation=False)


# def generate_str_seq(fullseq):
#     letters = ['A','B','C','D','E','F','G','H','I','J','K']
#     str_seq = ''.join([letters[int(fullseq[s,0,0]] for s in range(0,2000)])
#     return str_seq
