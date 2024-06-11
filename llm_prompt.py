# llm prompt generation

import sys
import numpy as np
import pandas as pd
from Learning import *
from CG1 import *
from chunks import *
import numpy as np
from scipy import stats


def acc_eval1d(img_seq, gt):
    """ Compare the accuracy between an imaginary sequence and a ground truth sequence """
    l = 0
    for i in range(0, img_seq.shape[0]):
        if img_seq[i,0,0] == gt[i,0,0]:
            l = l + 1
    return l/img_seq.shape[0]



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
        inst_seq.append(ka[int(seq[i,0,0])-1])
    return inst_seq

def calculate_prob(chunk_record, cg):
    p = 1
    for key in list(chunk_record.keys()):  # key is the encoding time
        p = p * cg.chunks[chunk_record[key][0][0]].count / np.sum([item.count for item in list(cg.chunks.values())])
    return p

def convert_observation_sequence(seq):
    Observations = []
    T, H, W = seq.shape
    for t in range(0, T):
        for h in range(0, H):
            for w in range(0, W):
                v = seq[t, h, w]
                if v != 0:
                    Observations.append((t, h, w, int(seq[t, h, w])))
    return Observations, T


def get_instruction_sequence(trial, df, sub, seql = 12):
    # obtain instruction sequence up to the current trial
    ins_seq = df[(df['ID'] == sub)].iloc[(1 - 1) * seql:trial * seql, :][
        'instructioncollect']
    ins_list = ins_seq.tolist()
    subj_recall = df[(df['ID'] == sub)].iloc[(1 - 1) * seql:trial * seql, :][
        'recallcollect']
    keyassignment = list(df[(df['ID'] == sub)].iloc[(1 - 1) * seql:trial * seql, :][
                             'keyassignment'])[0]
    condition = list(df[(df['ID'] == sub)].iloc[(1 - 1) * seql:trial * seql, :][
                         'condition'])[0]
    block = list(df[(df['ID'] == sub)].iloc[(1 - 1) * seql:trial * seql, :][
                     'blockcollect'])[0]
    trialRT = sum(list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                     'timecollect'])[0:12])
    proj_seq = convert_sequence(list(ins_seq), keyassignment)
    proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
    return proj_seq, condition, trialRT

def simonsaysex2():
    df = pd.read_csv('/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/data.csv')

    # dataframe that records trialwise accuracy
    dfm = {}  # model dataframe
    dfm['blockcollect'] = []
    dfm['ID'] = []
    dfm['condition'] = []
    dfm['correctcollect'] = []
    dfm['p'] = []
    dfm['trialcollect'] = []

    column_names = ['ID', 'condition', 'keyassignment', 'recallcollect', 'instructioncollect','trialcollect', 'correctcollect']
    dfs = pd.DataFrame(columns=column_names)  # Note that there are now row data inserted.

    seql = 12
    len_train = 40
    len_test = 8


    for sub in np.unique(list(df['ID'])):
        # initialize chunking part with specified parameters
        cg = CG1(DT=0.1, theta=0.996)
        for trial in range(1, len_train + 3 * len_test + 1):
            ins_seq = df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                'instructioncollect']
            ins_list = ins_seq.tolist()
            subj_recall = df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                'recallcollect']
            keyassignment = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                                     'keyassignment'])[0]
            condition = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                                 'condition'])[0]
            block = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                             'blockcollect'])[0]
            inst_seq = [''.join(item[1]) for item in list(ins_seq)]
            proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
            # enable_abstraction = condition == 'm1'
            cg, chunkrecord = hcm_learning(proj_seq, cg, abstraction=condition == 'm1')  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            seq_p = calculate_prob(chunkrecord, cg)

            recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])

            model_recall_seq = convert_sequence_backward_to_key(recalled_seq,keyassignment)

            p_seq = np.prod(ps)  # evaluate the probability of a sequence
            p_seq = seq_p
            # if condition != 'ind' and sub != 1:
            if trial == 1:
                print()
            if trial == 5:
                print()
            if trial == 30:
                print()

            if trial == 40:
                print()

            if trial == 41:
                print()

            if trial == 64:
                print()

            dfm['blockcollect'].append(block)
            dfm['ID'].append(sub)
            dfm['condition'].append(condition)
            dfm['correctcollect'].append(acc_eval1d(recalled_seq, proj_seq))
            dfm['p'].append(p_seq)
            dfm['trialcollect'].append(trial)


            for i in range(0, 12):
                if model_recall_seq[i] == ins_list[i]:
                    correctcollect = 1
                else: correctcollect = 0
                dfs = dfs.append({'ID': sub,
                                'keyassignment': keyassignment,
                                'condition': condition,
                                'recallcollect': model_recall_seq[i],
                                'trialcollect': trial,
                                'instructioncollect': ins_list[i],
                                'correctcollect': correctcollect}, ignore_index=True)

    dfm = pd.DataFrame.from_dict(dfm)
    csv_save_directory = '/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/simulation_data_model_transition_recall_3.csv'
    dfm.to_csv(csv_save_directory, index=False, header=True)


    dfs_csv_save_directory = '/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/simulation_data_model_transition_recall_individualkey_3.csv'
    dfs.to_csv(dfs_csv_save_directory, index=False, header=True)
    return



def get_chunk_record(arayseq, cg):
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)# update initial parameters
    seq, seql = convert_observation_sequence(arayseq[:, :, :])# loaded with the 0th observation
    cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
    return chunkrecord


def spearmancorr(x,y):
    res = stats.spearmanr(x, y)
    return res.statistic, res.pvalue


df = pd.read_csv('/Users/swu/Desktop/research/motif_learning/data/simonsays_ex2/data.csv')

# dataframe that records trialwise accuracy
dfm = {}  # model dataframe
dfm['blockcollect'] = []
dfm['ID'] = []
dfm['condition'] = []
dfm['correctcollect'] = []
dfm['p'] = []
dfm['trialcollect'] = []

column_names = ['ID', 'condition', 'keyassignment', 'recallcollect', 'instructioncollect','trialcollect', 'correctcollect']
dfs = pd.DataFrame(columns=column_names)  # Note that there are now row data inserted.

seql = 12
len_train = 40
len_test = 8

motif_population_instruction = {}
control_population_instruction = {}
import json



for sub in np.unique(list(df['ID'])):
    inst_seq = 'You are in a memory experiment, in this task, a kitty shows you a sequence, and you need to remember the sequence that the kitty shows you. '
    sub_list = []
    type = None
    # initialize chunking part with specified parameters
    cg = CG1(DT=0.1, theta=0.996)
    for trial in range(1, len_train + 3 * len_test + 1):
        trial_seq = ''
        ins_seq = df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
            'instructioncollect']
        ins_list = ins_seq.tolist()
        subj_recall = df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
            'recallcollect']
        keyassignment = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                                 'keyassignment'])[0]
        condition = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                             'condition'])[0]
        if trial == 1 and condition == 'm1': type = 'm1' # alter the instruction type here
        if trial == 1 and condition == 'ind': type = 'ind' # alter the instruction type here

        block = list(df[(df['ID'] == sub)].iloc[(trial - 1) * seql:trial * seql, :][
                         'blockcollect'])[0]

        inst_seq += ' the kitty shows you a sequence: '
        for item in list(ins_seq):
            inst_seq = inst_seq + item
            trial_seq = trial_seq + item
        inst_seq = inst_seq + '\n'
        sub_list.append(trial_seq)
    if type == 'm1': motif_population_instruction[str(sub)] = sub_list
    if type == 'ind': control_population_instruction[str(sub)] = sub_list

# file_path = str(sub) + '.txt'  # Name of the file to save the text in
# with open(file_path, 'w') as file:
#     file.write(inst_seq)
print(inst_seq)


import pickle
# Saving the dictionary to a pickle file
with open('instruction_data_for_llm_m1.pkl', 'wb') as pickle_file:
    pickle.dump(motif_population_instruction, pickle_file)
with open('instruction_data_for_llm_ind.pkl', 'wb') as pickle_file:
    pickle.dump(control_population_instruction, pickle_file)

# Loading the dictionary from a pickle file
with open('instruction_data_for_llm_ind.pkl', 'rb') as pickle_file:
    loaded_dict = pickle.load(pickle_file)

print(loaded_dict)


with open('instruction_data_for_llm_m1.json', 'w') as json_file:
    json.dump(motif_population_instruction, json_file)
with open('instruction_data_for_llm_ind.json', 'w') as json_file:
    json.dump(control_population_instruction, json_file)

#
# Read data back from a file
with open('instruction_data_for_llm_m1.json', 'r') as json_file:
    loaded_dict = json.load(json_file)

print(loaded_dict)