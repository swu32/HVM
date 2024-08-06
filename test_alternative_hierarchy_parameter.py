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

def test_random_graph_abstraction(generation = False):
    d = 30# depth increment parameter
    a = 10# alphabet size
    if generation:
        random_abstract_representation_graph(save=True, alphabet_size=a, depth=d, seql = 5000)

    # with open('random_abstract_sequence.npy', 'rb') as f:
    #     seq = np.load(f)
    for sz in depth_increment:
        openpath = './generative_sequences_different_parameters/random_abstract_sequence_a='+ str(a) + '_d=' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = 1000
        n_measure = 9
        n_run = 5#int(len(fullseq)/slice_sz)
        n_iter = 10
        datahcm = np.empty((n_run, n_iter, n_measure))
        datahvm = np.empty((n_run, n_iter, n_measure))
        i = 0
        for seq in slicer(fullseq, slice_sz):
            if i == n_run: break # just do 1 iteration
            cghvm = CG1(DT=0.1, theta=0.996)
            cghvm = hcm_markov_control_1(seq, cghvm, MAXit = n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            cghcm = CG1(DT=0.1, theta=0.996)
            cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit = n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            datahcm[i,:,:] = np.array(cghcm.learning_data)
            datahvm[i,:,:] = np.array(cghvm.learning_data)
            i = i + 1
        np.save('./data/hcm_fixed_support_set' + ' d = ' + str(sz) + '.npy', datahcm)
        np.save('./data/hvm_fixed_support_set' + ' d = ' + str(sz) + '.npy', datahvm)
        plot_average_model_learning_comparison(datahcm, datahvm, d = sz, savename = './data/fixed_support_set' + ' d = ' + str(sz) + '.png')
    return
