# Recovery analysis, associative learning model on human experiment
import sys
import pandas as pd
import seaborn as sns
import numpy as np
from Learning import *
from CG1 import *

def AL_experiment2(theta = 0.996, save_path = '/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/simulation_data_model_transition_recall_associative_learning_.csv', save_keys = False):
    df = pd.read_csv('/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/data.csv')
    dfm = {}  # model dataframe
    dfm['blockcollect'] = []
    dfm['ID'] = []
    dfm['condition'] = []
    dfm['correctcollect'] = []
    dfm['p'] = []
    dfm['trialcollect'] = []
    dfm['recall_likelihood'] = []  # the likelihood of subjects recalling the exact piece of sequence, given the representation learned so far

    column_names = ['ID', 'condition', 'recallcollect', 'instructioncollect', 'trialcollect', 'correctcollect']
    dfs = pd.DataFrame(columns=column_names)  # Note that there are now row data inserted.

    seql = 12
    len_train = 40
    len_test = 8

    def acc_eval1d(img_seq, gt):
        """ Compare the accuracy between an imaginary sequence and a ground truth sequence """
        l = 0
        for i in range(0, img_seq.shape[0]):
            if img_seq[i, 0, 0] == gt[i, 0, 0]:
                l = l + 1
        return l / img_seq.shape[0]

    def convert_sequence(seq, keyassignment):
        keyassignment = list(keyassignment)
        ka = [keyassignment[2], keyassignment[7], keyassignment[12], keyassignment[17], keyassignment[22],
              keyassignment[27]]
        proj_seq = [ka.index(item) + 1 for item in seq]
        return proj_seq

    def convert_sequence_backward_to_key(seq, keyassignment):
        keyassignment = list(keyassignment)
        inst_seq = []
        ka = [keyassignment[2], keyassignment[7], keyassignment[12], keyassignment[17], keyassignment[22],
              keyassignment[27]]
        for i in range(0, 12):
            inst_seq.append(ka[int(seq[i, 0, 0]) - 1])
        return inst_seq

    def calculate_prob(chunk_record, cg):
        p = 1
        for key in list(chunk_record.keys()):  # key is the encoding time
            p = p * cg.chunks[chunk_record[key][0][0]].count / np.sum([item.count for item in cg.chunks.values()])
        return p

    for sub in np.unique(list(df['ID'])):
        # initialize chunking part with specified parameters
        cg = CG1(DT=0.1, theta=theta)
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
            # print(proj_seq)
            proj_seq = np.array(proj_seq).reshape([-1, 1, 1])

            cg, chunkrecord = hcm_learning(proj_seq, cg, learn = True, chunk = False, abstraction = False)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])
            keypress_recalled_seq = convert_sequence_backward_to_key(recalled_seq, keyassignment)
            p_recall = np.prod(ps)  # evaluate the probability of a sequence
            ###### calculate the probability of the instruction sequences
            instr_seq_p = calculate_prob(chunkrecord, cg)


            dfm['blockcollect'].append(block)
            dfm['ID'].append(sub)
            dfm['condition'].append(condition)
            dfm['correctcollect'].append(acc_eval1d(recalled_seq, proj_seq))
            dfm['p'].append(p_recall)
            dfm['trialcollect'].append(trial)
            dfm['recall_likelihood'].append(instr_seq_p)

            ins_list = ins_seq.tolist()
            print(keypress_recalled_seq, ins_list)
            for i in range(0, 12):
                if keypress_recalled_seq[i] == ins_list[i]:
                    correctcollect = 1
                else:
                    correctcollect = 0
                dfs = dfs.append({'ID': sub,
                                  'condition': condition,
                                  'keyassignment': keyassignment,
                                  'recallcollect': keypress_recalled_seq[i],
                                  'trialcollect': trial,
                                  'instructioncollect': ins_list[i],
                                  'correctcollect': correctcollect}, ignore_index=True)

    dfm = pd.DataFrame.from_dict(dfm)
    csv_save_directory = save_path
    dfm.to_csv(csv_save_directory, index=False, header=True)

    if save_keys:
        dfs_csv_save_directory = '/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/simulation_data_model_transition_recall_associative_learning_individualkey_theta=' + str(theta) + '.csv'
        dfs.to_csv(dfs_csv_save_directory, index=False, header=True)

    return


