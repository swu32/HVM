# code to simulate participants' behavior in Simon Says experiment
import sys
import numpy as np
import pandas as pd
from Learning import *
from CG1 import *
from chunks import *
import numpy as np


def acc_eval1d(img_seq, gt):
    """ Compare the accuracy between an imaginary sequence and a ground truth sequence """
    l = 0
    for i in range(0, img_seq.shape[0]):
        if img_seq[i,0,0] == gt[i,0,0]:
            l = l + 1
    return l/img_seq.shape[0]

def simonsaysex2():
    df = pd.read_csv('/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/data.csv')

    dfm = {}  # model dataframe
    dfm['blockcollect'] = []
    dfm['ID'] = []
    dfm['condition'] = []
    dfm['correctcollect'] = []
    dfm['p'] = []
    dfm['trialcollect'] = []

    seql = 12
    len_train = 40
    len_test = 8

    def convert_sequence(seq, keyassignment):
        keyassignment = list(keyassignment)
        ka = [keyassignment[2], keyassignment[7], keyassignment[12], keyassignment[17], keyassignment[22],
              keyassignment[27]]
        proj_seq = [ka.index(item) + 1 for item in seq]
        return proj_seq

    def calculate_prob(chunk_record, cg):
        p = 1
        for key in list(chunk_record.keys()):  # key is the encoding time
            p = p * cg.chunks[chunk_record[key][0][0]].count / np.sum([item.count for item in cg.chunks])
        return p

    for sub in np.unique(list(df['ID'])):
        # initialize chunking part with specified parameters
        cg = CG1(DT=0.1, theta=0.996)
        for trial in range(1, len_train + 3 * len_test + 1):
            ins_seq = df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                'instructioncollect']
            subj_recall = df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                'recallcollect']
            keyassignment = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                                     'keyassignment'])[0]
            condition = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                                 'condition'])[0]
            block = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                             'blockcollect'])[0]
            proj_seq = convert_sequence(list(ins_seq), keyassignment)
            proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
            #enable_abstraction = condition == 'm1'
            cg, chunkrecord = hcm_learning(proj_seq, cg, abstraction=condition == 'm1')  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])
            p_seq = np.prod(ps)  # evaluate the probability of a sequence
            # if trial == 39:
            #     print()

            dfm['blockcollect'].append(block)
            dfm['ID'].append(sub)
            dfm['condition'].append(condition)
            dfm['correctcollect'].append(acc_eval1d(recalled_seq, proj_seq))
            dfm['p'].append(p_seq)
            dfm['trialcollect'].append(trial)

    dfm = pd.DataFrame.from_dict(dfm)
    csv_save_directory = '/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/simulation_data_model_transition_recall.csv'

    dfm.to_csv(csv_save_directory, index=False, header=True)
    return