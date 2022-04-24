# TODO: learning to classes and learning algorithm
# TODO: KL divergence plotting
# TODO: hierarhical graph structure plotting
# TODO: Test and reorganize the
from Hand_made_generative import *
from Generative_Model import *
from Learning import *
from plotting import *
from train import *
from model import *
from dataset import *
from CG1 import *
from chunks import *
import numpy as np
import PIL as PIL
from PIL import Image
import os
from time import time


from avaprocess import *
from chunks import *


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

                imagined_seq = NN_testing(seq)
                imagined_seq = np.array(imagined_seq).reshape([len(imagined_seq), 1, 1])
                klnn = evaluate_KL_compared_to_ground_truth(imagined_seq, cg_gt.M, Chunking_Graph(DT=0, theta=1))

                # take in data:
                df['N'].append(n)
                df['d'].append(depth)
                df['kl'].append(kl)
                df['type'].append('ck')

                df['N'].append(n)
                df['d'].append(depth)
                df['kl'].append(klnn)
                df['type'].append('nn')

    df = pd.DataFrame.from_dict(df)
    df.to_pickle('KL_rational_learning_N')  # where to save it, usually as a .pkl
    return df


def transfer_pre_training(cg_train, n_train=3000):
    """cg_train: chunking graph define to generate transfer sequences
        n_train: use a sequence of length 3000 to train a model"""
    training_seq = generate_hierarchical_sequence(cg_train.M, s_length=n_train)
    cg = Chunking_Graph(DT=0.1, theta=0.96)  # naive model with no knowledge about the sequence
    cg = learn_stc_classes(training_seq, cg)
    return cg

def transferinterferenceexperiment():

    def training_session(graph_pipeline):
        # TODO: please make sure that the graph tuple is indeed updated with the trained graph
        for g in range(0, len(graph_pipeline['transfer'])):
            graphtuple = graph_pipeline['transfer'][g]
            _, cg_train = graphtuple
            cg_trained = transfer_pre_training(cg_train)
            graph_pipeline['transfer'][g] = graphtuple + (cg_trained,)

        for g in range(0, len(graph_pipeline['nonoverlap'])):
            graphtuple = graph_pipeline['nonoverlap'][g]
            _, cg_train = graphtuple
            cg_trained = transfer_pre_training(cg_train)
            graph_pipeline['nonoverlap'][g] = graphtuple + (cg_trained,)

        for g in range(0, len(graph_pipeline['semioverlapping'])):
            graphtuple = graph_pipeline['semioverlapping'][g]
            _, _, cg_train = graphtuple
            cg_trained = transfer_pre_training(cg_train)
            graph_pipeline['semioverlapping'][g] = graphtuple + (cg_trained,)

        return graph_pipeline

    def transfer_train_measure_KL(cg_trained,cg_test,training_sequence):
        cg_trained = learn_stc_classes(training_sequence, cg_trained)
        imagined_seq_trained = cg_trained.imagination(1000, sequential=True, spatial=False, spatial_temporal=False)
        kl = evaluate_KL_compared_to_ground_truth(imagined_seq_trained, cg_test.M, Chunking_Graph(DT=0, theta=1))
        return kl

    def test_and_KL_measurement(graph_pipeline,training_sequence,df,N, cg_test):
        naive_learner = Chunking_Graph(DT=0.1, theta=0.96)
        kl = transfer_train_measure_KL(naive_learner, cg_test, training_sequence)
        df['type'].append('naive')
        df['N'].append(N)
        df['kl'].append(kl)
        df['delete'].append(0)
        df['inferfere'].append(0)
        df['n_s'].append(0)
        df['n_d'].append(0)

        for graphtuple in graph_pipeline['transfer']:
            d, _, cg_trained = graphtuple
            kl = transfer_train_measure_KL(cg_trained,cg_test,training_sequence)
            df['type'].append('transfer')
            df['N'].append(N)
            df['kl'].append(kl)
            df['delete'].append(d)
            df['inferfere'].append(0)
            df['n_s'].append(0)
            df['n_d'].append(0)

        for graphtuple in graph_pipeline['nonoverlap']:
            d, _, cg_trained = graphtuple
            cg_trained = learn_stc_classes(training_sequence, cg_trained)
            kl = transfer_train_measure_KL(cg_trained,cg_test,training_sequence)
            df['type'].append('nonoverlap')
            df['N'].append(N)
            df['kl'].append(kl)
            df['delete'].append(0)
            df['inferfere'].append(d)
            df['n_s'].append(0)
            df['n_d'].append(0)

        for graphtuple in graph_pipeline['semioverlapping']:
            n_s,n_d,_, cg_trained = graphtuple
            kl = transfer_train_measure_KL(cg_trained,cg_test,training_sequence)
            df['type'].append('semioverlapping')
            df['N'].append(N)
            df['kl'].append(kl)
            df['delete'].append(0)
            df['inferfere'].append(0)
            df['n_s'].append(n_s)
            df['n_d'].append(n_d)
        return df


    ################ Testing Transfer and Interference Effect #################
    n_sample = 10
    df = {}
    df['type'] = []
    df['N']=[]
    df['kl']=[]
    df['delete']=[]
    df['inferfere']=[]
    df['n_s']=[]
    df['n_d']=[]
    # in each sample, a different graph is generated, should we do that?
    graph_pipeline = transfer_graphs_generator()
    for k in range(0, n_sample):
        trained_graph_pipeline = training_session(graph_pipeline)
        # use test graph to generate test_seqeunce
        cg_test = graph_pipeline['test']
        N_test = 3000  # time step 3000 to test the transfer effect
        test_sequence = generate_hierarchical_sequence(cg_test.M, s_length=N_test)

        N_interval = 500
        N = 0
        for i in range(0, 6):
            N = N + N_interval
            test_sequence_segment = test_sequence[i*N_interval:(i+1)*N_interval,:,:]
            df = test_and_KL_measurement(trained_graph_pipeline, test_sequence_segment, df, N, cg_test)
        # measure KL divergence after training
        # after training, test performance between the trained model and naive model on producing the geneative-like
        # structure on the * transfer complex structure *
        # track learning progress as the model parses through the sequence
    df = pd.DataFrame.from_dict(df)
    df.to_csv('TransferExperiment')
    return df


def NN_testing(sequence):
    '''Input: sequence of a certain size that the NN is used then to train'''
    ################ doing neural network testing here #############
    # convert the sequence into lists

    # Ns = np.arange(50, 3000, 50)# the length of sequence decided to show neural networks
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--sequence-length', type=int, default=3)
    args = parser.parse_args()

    dataset = Dataset(sequence, args)
    model = Model(dataset)

    train(dataset, model, args)
    imaginary_sequence = predict(dataset, model)
    return imaginary_sequence

def c3_RNN():
    # need a list of prediction and output probability.
    # train until the next mistake.
    '''Compare neural network behavior with human on chunk prediction'''
    sequence = np.array(generateseq('c3', seql=800)).reshape((800,1,1))
    sequence[0,:,:] = 0
    #sequence = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--sequence-length', type=int, default=5)
    args = parser.parse_args()
    start = 200
    predicted_seq = []#list(sequence[0:start].flatten())
    prob = []#[0.25]*start
    idx = start
    while idx < 800:
        dataset = Dataset(sequence[0:idx], args)# use all of the past dataset to train
        model = Model(dataset)
        train(dataset, model, args) # train another model from scratch.
        predicted_words, ps = predict(dataset, model, next_words=20, words=list(sequence[max(idx-20,0):idx,:,:].flatten()))
        step = 0
        while predicted_words[step] == sequence[idx+step] and step <20:
            print(predicted_words[step], sequence[idx + step])
            predicted_seq = predicted_seq + [predicted_words[step]]
            prob = prob + [ps[step]]
            step = step + 1
            if idx+step>=800:
                break

        idx = idx + step

    df = {}
    df['seq'] = predicted_seq
    df['prob'] = prob

    # df = pd.DataFrame.from_dict(df)
    # df.to_pickle('c3_RNN.pkl')  # where to save it, usually as a .pkl
    with open('c3_RNN.npy', 'wb') as f:
        np.save(f, [predicted_seq, prob])
    return


def wikitext():
    # evaluate preplexity on wikitext
    # TODO: train HCM on wikitext's training set,
    # TODO: validate HCM on wikitext's testing set
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
    gifarray = readGif('/Users/shuchenwu/Desktop/research/chunking/code/gif_data/octo_25.gif')
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

    # cg = Chunking_Graph(DT=0.1, theta=0.98)  # initialize chunking part with specified parameters
    # cg = learn_stc_classes(totalseq, cg)
    cg = CG1(DT=0.1, theta=0.96)  # initialize chunking part with specified parameters
    cg = hcm_learning(totalseq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    cg.convert_chunks_in_arrays()
    print(totalseq.shape)
    # transform each chunk into gif array
    K = sorted(cg.chunks, key=lambda x: x.volume, reverse=True)  # for decreasing order
    #K = sorted(cg.M.items(), key=lambda x: np.sum(tuple_to_arr(x[1])>0), reverse=True)  # for decreasing order
    for k in range(0, len(K)):
        test_chunk = K[k].arraycontent
        for p in range(0, test_chunk.shape[0]):
            gif_chunk = np.zeros((25, 25, 4))
            for i in range(0, 25):
                for j in range(0, 25):
                    gif_chunk[i, j, :] = np.array(colormap[int(test_chunk[p, i, j])])

            gif_chunk = (255.0 / gif_chunk.max() * (gif_chunk - gif_chunk.min())).astype(np.uint8)
            im = Image.fromarray(gif_chunk)
            name = '/Users/shuchenwu/Desktop/research/chunking/code/images/squid/' + str(k) + '|-' + str(p) + '.png'
            im.save(name)

    return


def fmri():
    import numpy as np
    with open('/Users/swu/Downloads/behaviorialintegratedtimeseries.npy', 'rb') as f:
        whole_time_series = np.load(f)
    subject_learned_chunk = []
    for i in range(0, whole_time_series.shape[0]):
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

    for ID in range(0, 30): # across 30 runs
        hcm_chunk_record[ID] = []
        seq = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
        cg = CG1(DT=0.0, theta=0.96)  # initialize chunking part with specified parameters
        cg, chunkrecord = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        for time in list(chunkrecord.keys()):
            df['time'].append(int(time))
            ckidx = chunkrecord[time][0][0]
            df['chunksize'].append(cg.chunks[ckidx].volume)
            df['ID'].append(ID)
            chunk = get_chunk_list(cg.chunks[ckidx])
            hcm_chunk_record[ID].append(chunk)
            if 0 in chunk:
                print('')


    with open('HCM_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)

    with open('HCM_chunk.pkl', 'wb') as f:
        pickle.dump(hcm_chunk_record, f)

    seq_prediction = cg.prediction(seq)  # given a sequence, predict the next several elements given the previous

    return seq_prediction

def calcium_imaging():

    ################# Calcium Imagining Processing #################
    new_num_arr = np.load('../rasteroutput.npy')  # load
    T = 0.2
    new_num_arr[new_num_arr > T] = 1
    new_num_arr[new_num_arr < T] = 0
    new_num_arr = new_num_arr.T
    seq = new_num_arr.reshape([23185, 100, 1])

    cg = CG1(DT=0.1, theta=0.96)  # initialize chunking part with specified parameters
    cg = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    cg.convert_chunks_in_arrays()
    data = []
    for ck in cg.chunks:
        data.append(ck.arraycontent)
    np.save('calcium_imaging_data.npy', data, allow_pickle=True)


    data = []
    for ck in list(cg.chunks):
        if len(ck.content)>1:
            arraycontent = np.zeros((30, ck.H, ck.W))
            print(arraycontent.shape)
            for t, i, j, v in ck.content:
                arraycontent[t, i, j] = v
            data.append(arraycontent)
    np.save('calcium_imaging_data.npy', data, allow_pickle=True)
    calcium_imaging_data = np.load('calcium_imaging_data.npy', allow_pickle=True)
    # TODO： chunk identification can speed up with parent and children chunks, but since all chunks are scanned one by one it takes a long time.
    # TODO： sequence deletion can also take a long time, as time index is refactored.
    return


def previous_experiments_in_main():
    # c3_RNN()
    # cg_gt = generative_model_random_combination(D=5, n=5)
    # cg_gt = to_chunking_graph(cg_gt)
    # seq = generate_hierarchical_sequence(cg_gt.M, s_length=1000)
    # imagined_seq = NN_testing(seq)
    # fmri()
    # c3_chunk_learning()

    ################## Generative Model ################
    cggt = generative_model_random_combination(D=4, n=4)
    cggt = to_chunking_graph(cggt)
    seq = generate_random_hierarchical_sequence(cggt.M, s_length=500)
    cg = Chunking_Graph(DT=0.01, theta=1)  # initialize chunking part with specified parameters
    cg = learn_stc_classes(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    # one dimensional rational chunking
    learned_M, _, _, _ = partition_seq_hastily(seq, list(cggt.M.keys()))
    print(learned_M)
    # cg_gt = hierarchy1d()  # one dimensional chunks
    cg = Chunking_Graph(DT=0, theta=1)  # initialize chunking part with specified parameters
    cg = rational_chunking_all_info(seq, cg, maxit=3)

    ################# Debug and Intuitivize Spatial Temporal Chunking ############
    cg = Chunking_Graph(DT=0.01, theta=1)  # initialize chunking part with specified parameters
    cg = learn_stc_classes(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    ################ Debug Squid moving Gif #########################
    squidgifmoving()

    ################ Testing Transfer   and Intereference Graph Generation Techniques.
    df = transferinterferenceexperiment()
    ################# Measure KL ##############
    measure_KL()

    ################# Transfer and Interference and their KL Divergence plots ##############

    cg_trained = Chunking_Graph(DT=0.005, theta=0.996)

    cg_original = transfer_original()

    n_train = 3000

    # train the model on the facilitating sequence:
    training_seq = generate_hierarchical_sequence(cg_original.M, s_length=n_train)
    cggt = generative_model_random_combination(D=4, n=4)
    cggt = to_chunking_graph(cggt)
    print(cggt.M)
    seq = generate_random_hierarchical_sequence(cggt.M, s_length=5000)

    cg_trained = learn_stc_classes(seq, cg_trained)

    ################# experiment to measure KL divergence ###############
    Ns = np.arange(100, 3000, 100)
    measure_KL(Ns)
    # TODO: plot temporal chunks and its components
    cg_gt = hierarchy1d()
    n = 2000
    seq = generate_hierarchical_sequence(cg_gt.M, s_length=n)
    cg = Chunking_Graph(DT=0, theta=1)  # initialize chunking part with specified parameters
    cg = learn_stc_classes(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    return

def main():
    fmri()


    pass

if __name__ == "__main__":
    main()

