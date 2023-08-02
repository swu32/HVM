from Hand_made_generative import *
from Generative_Model import *
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
from simonsays import *

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


def p_RNN(trainingseq,testingseq):
    # need a list of prediction and output probability.
    # train until the next mistake.
    '''Compare neural network behavior with human on chunk prediction'''
    sequence = np.array(trainingseq).reshape((-1,1,1))
    sequence[0:5,:,:] = np.array([0,1,2,3,4]).reshape((5,1,1))
    #sequence = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--sequence-length', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    args = parser.parse_args()


    dataset = Dataset(sequence, args)  # use all of the past dataset to train
    model = Model(dataset)
    train(dataset, model, args)  # train another model from scratch.

    start = 10
    prob = [0.25]*start
    testsequence = np.array(testingseq).reshape((-1,1,1))
    for idx in range(start, testsequence.shape[0]):
        pre_l = 10
        p_next = evaluate_next_word_probability(model, testsequence[idx], words=list(testsequence[max(idx-pre_l,0):idx,:,:].flatten()))
        prob.append(p_next[0][0])

    return prob


def wikitext():
    # evaluate preplexity on wikitext
    # TODO: train HCM on wikitext's training set,
    # TODO: validate HCM on wikitext's testing set, including the perplexity score
    return

def NN_data_record():
    ################# Training Neural Networks to Compare with Learning Sequence ###########

    df = {}

    df['N'] = []
    df['klnn'] = []
    n_sample = 5  # taking 10 samples for each of the N specifications.
    Ns = np.arange(50, 3000, 50)

    cg_gt = generative_model_random_combination(D=3, n=5)
    cg_gt = to_chunking_graph(cg_gt)

    for i in range(0, n_sample):
        # Ns = np.arange(100,3000,100)

        for j in range(0, len(Ns)):
            n = Ns[j]
            seq = generate_hierarchical_sequence(cg_gt.M, s_length=n)
            print(len(seq))
            imagined_seq = NN_testing(seq)
            imagined_seq = np.array(imagined_seq).reshape([len(imagined_seq),1,1])
            kl = evaluate_KL_compared_to_ground_truth(imagined_seq, cg_gt.M, Chunking_Graph(DT=0, theta=1))
            df['N'].append(n)
            df['klnn'].append(kl)
            print({'kl is ': kl})

    df = pd.DataFrame.from_dict(df)
    df.to_pickle('../KL_neural_network_N')  # where to save it, usually as a .pkl
    return

def learn_hunger_game():
    ############### Learning Demonstration on Real World Datasets ###############
    file1 = list(open('text_data.txt').read())

    file1 = [s.lower() for s in file1]
    # convert the list of characters into list of integers, with empty space being the 0 integer.
    character_counts = Counter(file1)
    unique_char = sorted(character_counts, key=character_counts.get, reverse=True)
    unique_char.insert(0,'lll')
    index_to_char = {index: char for index, char in enumerate(unique_char)}
    char_to_index = {char: index for index, char in enumerate(unique_char)} # empty spaces are 0 as the most are empty spaces.

    seq_int = [char_to_index[w] for w in file1] # convert a book into a sequence of integers for processing

    # convert seq_int to interpretable sequences
    language_sequence = np.array(seq_int).reshape([len(seq_int),1,1])

    print(language_sequence)
    cg = Chunking_Graph(DT=0.00001, theta=0.999996)  # initialize chunking part with specified parameter
    DATA = {}
    DATA['N'] = []
    DATA['chunk learned'] = []
    # show that the words learned are different between different stages of learning
    intr = 1000
    for i in range(200,400):
        # interval of 50000 words
        cg = learn_stc_classes(language_sequence[intr*i:intr*(i+1),:,:],cg)
        print('after learning this number of iterations ', intr*i)
        chunks = []
        for chunk in list(cg.M.keys()):
            ckarr = tuple_to_arr(chunk)
            backwardindex = list(np.ravel(ckarr))
            seq_word = [index_to_char[i] for i in backwardindex]
            print('learned chunk: ', "".join(seq_word))
            chunks.append("".join(seq_word))
        DATA['chunk learned'].append(chunks)
        DATA['N'].append(intr*(i+1))
        print( 'start next round')
    return


def trainonava():
    # single person action sequence, mix together space and time, but annotated single person per sequence size
    # Need to interpret sequence chunks back to meaningful space

    labelmap, class_ids = read_labelmap("./data/label.txt")
    seq = read_csv('./data/ava_train_v2.2.csv', class_whitelist=None, capacity=0)

    # first step: train
    # second step: translate what has been learned
    # multiple person can be at the same time action sequence

    pass



# helper function to read gif
def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)

    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if asNumpy is false, a list if PIL images.

    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: ' + str(filename))

    # Load file using PIL
    pilIm = PIL.Image.open(filename)
    pilIm.seek(0)

    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert()  # Make without palette
            a = np.asarray(tmp)
            if len(a.shape) == 0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell() + 1)
    except EOFError:
        pass

    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:
            images.append(PIL.Image.fromarray(im))

    # Done
    return images




def squidgifmoving():
    def process_gif_data():
        gifarray = readGif('/Users/swu/Desktop/research/chunking/code/gif_data/octo_25.gif')
        # this function loads gifs into a list of np arrays
        T = len(gifarray)
        print(T)

        # find a unique combination of colors, assign them as interger, the blue color should be 0
        colormap = []
        cm = 0
        animseq = np.zeros((T, 25, 25))
        for t in range(0, T):
            thisarray = gifarray[t]
            for i in range(0, 25):
                for j in range(0, 25):
                    if tuple(thisarray[i, j, :]) not in colormap:
                        colormap.append(tuple(thisarray[i, j, :]))
                        cm = cm + 1
                    animseq[t, i, j] = colormap.index(tuple(thisarray[i, j, :]))

        # need a nparray to gif converter for the learned representations by the model.
        R = 100  # the number of repetition
        totalseq = np.zeros((T * R, 25, 25))
        # make the sequence repeat
        for i in range(0, R):
            totalseq[i * T:(i + 1) * T, :, :] = animseq
        totalseq = totalseq.astype(int)
        # cg = Chunking_Graph(DT=0.1, theta=0.98)  # initialize chunking part with specified parameters
        # cg = learn_stc_classes(totalseq, cg)
        return totalseq

    totalseq = process_gif_data()
    cg = CG1(DT=0.1, theta=0.996)
    # cg,chunkrecord = hcm_rational_v1(totalseq[:10,:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    # initialize chunking part with specified parameters
    cg,chunkrecord = hcm_rational(totalseq[:10,:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    cg.convert_chunks_in_arrays()
    print(totalseq.shape)
    # transform each chunk into gif array
    # K = sorted(cg.chunks, key=lambda x: x.volume, reverse=True)  # for decreasing order
    #K = sorted(cg.M.items(), key=lambda x: np.sum(tuple_to_arr(x[1])>0), reverse=True)  # for decreasing order
    for k in range(0, len(cg.chunks)):
        # test_chunk = K[k].arraycontent
        test_chunk = cg.chunks[k].arraycontent
        for p in range(0, test_chunk.shape[0]):
            gif_chunk = np.zeros((25, 25, 4))
            for i in range(0, 25):
                for j in range(0, 25):
                    gif_chunk[i, j, :] = np.array(colormap[int(test_chunk[p, i, j])])

            gif_chunk = (255.0 / gif_chunk.max() * (gif_chunk - gif_chunk.min())).astype(np.uint8)
            im = Image.fromarray(gif_chunk)
            name = '/Users/swu/Desktop/research/chunking/code/images/squid/' + str(k) + '|-' + str(p) + '.png'
            im.save(name)

    return


def fmri():
    import numpy as np
    with open('/Users/swu/Documents/MouseHCM/HSTC/fmri_timeseries/timeseries.npy', 'rb') as f:
        whole_time_series = np.load(f)
    subject_learned_chunk = []
    for i in range(0, whole_time_series.shape[0]):
        time_series = whole_time_series[i,:,:]
        seq = time_series.astype(int).reshape(time_series.shape + (1,))
        cg = CG1(DT=0.1, theta=1.0)  # initialize chunking part with specified parameters
        cg, chunkrecord = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        cg.save_graph(name = 'subject' + str(i), path = './fmri_chunk_data/')
        # reparse the sequence, using the biggest chunks
        cg.reinitialize()
        cg, chunkrecord = hcm_learning(seq, cg, learn = False)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

        # store chunks learned by cg
        learned_chunk = []
        for ck in cg.chunks:
            # record all the chunks
            ck.to_array()
            chunk_array = ck.arraycontent
            freq = ck.count
            learned_chunk.append((chunk_array, freq))
        subject_learned_chunk.append([learned_chunk, chunkrecord])

    with open('./fmri_chunk_data/fmri_learned_chunks.npy', 'wb') as f:
        np.save(f, subject_learned_chunk)

    return



def fmri_reshuffle_run():
    import numpy as np
    with open('/Users/swu/Downloads/behaviorialintegratedtimeseries.npy', 'rb') as f:
        whole_time_series = np.load(f)
    subject_learned_chunk = []
    for i in range(0, n_shuffle):
        # shuffle shuffle shuffle
        # find big chunks, and their index
        # find out whether
        time_series = whole_time_series[i,:,:]
        seq = time_series.astype(int).reshape(time_series.shape + (1,))
        cg = CG1(DT=0.1, theta=1.0)  # initialize chunking part with specified parameters
        cg, chunkrecord = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        cg.save_graph(name = 'subject' + str(i), path = './fmri_chunk_data/')
        # store chunks learned by cg
        learned_chunk = []
        for ck in cg.chunks:
            # record all the chunks
            ck.to_array()
            chunk_array = ck.arraycontent
            freq = ck.count
            learned_chunk.append((chunk_array, freq))
        subject_learned_chunk.append([learned_chunk, chunkrecord])

    with open('./fmri_chunk_data/fmri_learned_chunks.npy', 'wb') as f:
        np.save(f, subject_learned_chunk)

    return


def visual_chunks():
    # visual chunks
    cg_gt = compositional_imgs()
    n = 2000
    seq = generate_hierarchical_sequence(cg_gt.M, s_length=n)
    cg = CG1(DT=0.1, theta=0.96)  # initialize chunking part with specified parameters
    cg = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    cg.convert_chunks_in_arrays()
    cg.save_graph()
    return

def c3_chunk_learning():
    def get_chunk_list(ck):
        #print(np.array(list(ck.content)))
        T = int(max(np.array(list(ck.content)).reshape([-1,4])[:, 0])+1)
        chunk = np.zeros([T],dtype=int)
        for t,_,_, v in ck.content:
            print(ck.content, chunk.size, T)
            chunk[t] = v
        for item in list(chunk):
            if item == 0:
                print('')
        return list(chunk)

    import pickle
    ''' save chunk record for HCM learned on behaviorial data '''
    df = {}
    df['time'] = []
    df['chunksize'] = []
    df['ID'] = []

    hcm_chunk_record = {}

    for ID in range(0, 50): # across 30 runs
        hcm_chunk_record[ID] = []
        seq = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
        cg = CG1(DT=0.0, theta=0.92)  # initialize chunking part with specified parameters
        cg, chunkrecord = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        for time in list(chunkrecord.keys()):
            df['time'].append(int(time))
            ckidx = chunkrecord[time][0][0]
            df['chunksize'].append(cg.chunks[ckidx].volume)
            df['ID'].append(ID)
            chunk = get_chunk_list(cg.chunks[ckidx])
            hcm_chunk_record[ID].append(chunk)


    with open('HCM_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)

    with open('HCM_chunk.pkl', 'wb') as f:
        pickle.dump(hcm_chunk_record, f)

    return

def evaluate_perplexity(data, chunkrecord):
    #TODO: convert chunkrecord into sequence of probability
    p = []
    n_ck = 0
    for t in range(0, len(data)):
        if t in list(chunkrecord.keys()):
            freq = chunkrecord[t][0][1]
            n_ck = n_ck + 1
            p.append(freq/n_ck)
        else: # a within-chunk element
            p.append(1)
    perplexity = 2**(-np.sum(np.log2(np.array(p)))/len(p))

    return perplexity


def Wikitext2():
    # load data
    # train on chunking model
    # evaluate preplexity
    path = '/Users/swu/Documents/MouseHCM/HSTC/wikitext-2/'
    corpus = Corpus(path)

    train_data = np.array(corpus.train).reshape([-1, 1, 1])
    val_data = np.array(corpus.valid).reshape([-1, 1, 1])
    test_data = np.array(corpus.test).reshape([-1, 1, 1])
    # train, val, and test are 20:1:1
    unit_size = 50
    for i in range(0, 1000):# 100 times the unitsize
        cg = CG1(DT=0.00001, theta=0.99998)  # initialize chunking part with specified parameters
        cg, chunkrecord_train = hcm_learning(train_data[20*unit_size*i: 20*unit_size*(i+1),:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        cg, chunkrecord_test = hcm_learning(test_data[unit_size*i: unit_size*(i+1),:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        perplexity_test = evaluate_perplexity(test_data[unit_size*i: unit_size*(i+1),:,:], chunkrecord_test)
        print('test perplexity is ', perplexity_test)
        cg, chunkrecord_val = hcm_learning(val_data[unit_size*i: unit_size*(i+1),:,:], cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        perplexity_val = evaluate_perplexity(val_data[unit_size*i: unit_size*(i+1),:,:], chunkrecord_val)
        print('validation perplexity is ', perplexity_val)
        print(i)
        # should run this code on the cluster, and record the perplexity for testing and validation with the number of epochs,
        # at the moment, infeasible to run this code on PC with limied computing power.
        # also, need to integrate chunk deletion mechanism.
    # evaluate the probability of the test set
    # evaluate the sum of log probability across n observations
    # divide by N
    # remove the log by exponentiating
    return




def simonsaystransfer():
    # for the m1 and m2 subjects, you can characterize the distance from m1 - m2 and from m2 - m1.
    # this distance is the same for each m1 and each m2 participants
    # the distance is how much more difficult is it to learn m2 on top of m1, and vice versa
    # correlate this distance with drop in performance for each participant
    # see which distance is more similar to both groups
    pass

    def evalhcmdifficulty():
        # train m1 until convergence, and see how many iteration is needed to arrive at m2.
        return dm1m2, dm2m1

    def evaleditdistance():
        return E1, E2


def m1_m2():
    pass

    m1 = np.array([1,2,2,2, 2,2,1,1, 1,1,2,1]).reshape([-1, 1, 1])
    m2 = np.array([1,1,1,2, 2,1,1,2, 2,2,2,1]).reshape([-1, 1, 1])
    cgm1m2 = CG1(DT=0.1, theta=0.996)

    learned = False
    nrep = 0 # the number of repetition
    while ~learned:
        nrep = nrep + 1
        cgm1m2, chunkrecord = hcm_learning(m1, cgm1m2)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        if m1 in cgm1m2.chunks:
            learned = True



def simonsays():
    df = pd.read_csv('/Users/swu/Desktop/research/motif_learning/data/simonsays_ed/data.csv')

    dfm = {}  # model dataframe
    dfm['blockcollect'] = []
    dfm['ID'] = []
    dfm['condition'] = []
    dfm['correctcollect'] = []
    dfm['p'] = []
    dfm['trialcollect'] = []

    seql = 12
    len_train = 30
    len_test = 8
    def convert_sequence(seq):
        seq = list(seq)
        x = seq[0]
        proj_seq = [] # pause
        for item in seq:
            if item == x:
                proj_seq.append(1)
            else:
                proj_seq.append(2)
        return proj_seq
    def calculate_prob(chunk_record, cg):
        p = 1
        for key in list(chunk_record.keys()):# key is the encoding time
            p = p*cg.chunks[chunk_record[key][0][0]].count/np.sum([item.count for item in cg.chunks])
        return p

    for sub in np.unique(list(df['ID'])):
        # initialize chunking part with specified parameters
        cg = CG1(DT=0.1, theta=0.996)
        for trial in range(1, len_train + 3*len_test+ 1):
            ins_seq = df[(df['ID'] == sub)].iloc[(trial-1)*seql:trial*seql, :][
                'instructioncollect']
            condition = list(df[(df['ID'] == sub)].iloc[(trial-1)*seql:trial*seql, :][
                'condition'])[0]
            block = list(df[(df['ID'] == sub)].iloc[(trial-1)*seql:trial*seql, :][
                'blockcollect'])[0]
            proj_seq = convert_sequence(ins_seq)
            proj_seq = proj_seq  # display one time, and recall for another time
            proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
            cg, chunkrecord = hcm_learning(proj_seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            p_seq = calculate_prob(chunkrecord, cg)# evaluate the probability of a sequence
            recall_seq = cg.imagination1d(seql=12) # parse sequence using chunks and evaluate the chunk probability
            dfm['blockcollect'].append(block)
            dfm['ID'].append(sub)
            dfm['condition'].append(condition)
            dfm['correctcollect'].append(acc_eval1d(recall_seq, proj_seq))
            dfm['p'].append(p_seq)
            dfm['trialcollect'].append(trial)

    dfm = pd.DataFrame.from_dict(dfm)
    csv_save_directory = '/Users/swu/Desktop/research/motif_learning/data/simonsays/simulation_data_ed.csv'

    dfm.to_csv(csv_save_directory, index=False, header=True)

    return



def test_random_graph_abstraction():
    #cggt, seq = random_abstract_representation_graph(save=True)
    # with open('random_abstract_sequence.npy', 'rb') as f:
    #     seq = np.load(f)
    with open('sample_abstract_sequence.npy', 'rb') as f:
        seq = np.load(f)

    cg = CG1(DT=0.1, theta=0.996)
    cg = hcm_markov_control(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    return

def test_simple_abstraction():
    seq = simple_abstraction_I()
    cg = CG1(DT=0.1, theta=0.996)
    cg, chunkrecord = hcm_rational(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    return


def parseDNA():
    # TO use: seq = parseDNA()
    directory = "/Users/swu/Documents/MouseHCM/HSTC/genome_data/genome_assemblies_genome_fasta/ncbi-genomes-2022-12-01" \
                "/GCF_000005845.2_ASM584v2_genomic.txt"
    with open(directory) as f:
        STR = f.read()

    def split(word):
        return [char for char in word]

    IR = split(STR)
    seq = []
    for it in IR:
        if it == 'A':
            seq.append(1)
        if it == 'T':
            seq.append(2)
        if it == 'C':
            seq.append(3)
        if it == 'G':
            seq.append(4)
    seq = np.array(seq).reshape([len(seq), 1, 1])
    return seq

def test_motif_learning_experiment2():
    training_seq, testing_seq = exp2()
    #training_seq, testing_seq = exp2(control = True)

    cg = CG1(DT=0.1, theta=0.996)
    for i in range(0,40):
        proj_seq = training_seq[i]
        proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
        cg, chunkrecord = hcm_learning(proj_seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])
        if len(cg.variables.keys())>1:
            print()
        p_seq = np.prod(ps)  # evaluate the probability of a sequence
        print(p_seq)
    print('done with training')
    for i in range(0,24):
        proj_seq = testing_seq[i]
        proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
        cg, chunkrecord = hcm_learning(proj_seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])
        p_seq = np.prod(ps)  # evaluate the probability of a sequence
        print(recalled_seq, ps)

    return


def main():
    #seq = abstraction_illustration()
    #simonsaysex2()
    #test_motif_learning_experiment2()
    test_random_graph_abstraction()
    test_simple_abstraction()
    pass

if __name__ == "__main__":

    main()

