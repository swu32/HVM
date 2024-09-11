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


def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def test_random_graph_abstraction(generation = False):
    depth = 30# depth increment parameter
    alphabet_size = 10# alphabet size
    sequence_length = 5000
    if generation:
        for p_variable in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1.0]:
            random_abstract_representation_graph(save=True, alphabet_size=alphabet_size, depth=depth, seql=sequence_length, p_variable=p_variable)

    for p_variable in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1.0]:
        openpath = './generative_sequences_different_parameters/random_abstract_sequence_a='+ str(alphabet_size) + '_d='+ str(depth) + '_p_variable=' + str(p_variable) + '.npy'
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
            cghcm = hcm_markov_control_1(seq, cghcm, ABS=False, MAXit=n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            datahcm[i,:,:] = np.array(cghcm.learning_data)
            datahvm[i,:,:] = np.array(cghvm.learning_data)
            i = i + 1

        np.save('./data/hcm_fixed_support_set_a='+ str(alphabet_size) + '_d='+ str(depth) + '_p_variable=' + str(p_variable) + '.npy', datahcm)
        np.save('./data/hvm_fixed_support_set_a='+ str(alphabet_size) + '_d='+ str(depth) + '_p_variable=' + str(p_variable) + '.npy', datahvm)

        savename = './data/' +'_a='+ str(alphabet_size) + '_d='+ str(depth) + '_p_variable=' + str(p_variable) + '.png'

        # load ground truth model for comparison
        gt_load_name = './data/generative_hvm_a=' + str(alphabet_size) + '_d=' + str(depth) + '_p_variable=' + str(p_variable)+'.npy'
        plot_average_model_learning_comparison(datahcm, datahvm, d=depth, savename=savename, gt_load_name = gt_load_name)
    return

test_random_graph_abstraction()