from Learning import *

def simple_abstraction():
    """An abstraction structure with a variable and two chunks that shares the same variable"""
    seq = []
    l = 20
    c1 = [1,2,3]
    c2 = [2,1,2]
    c3 = [2, 1]
    for i in range(0,l):
        v1 = [c1, c2, c3][np.random.randint(0, 3)]
        c4 = [3] + v1 + [4]
        c5 = [1] + v1 + [2]
        seq = ([c4, c5][np.random.randint(0, 2)])
    return seq

def simple_abstraction_I():
    seql = 1000
    seq = np.zeros(shape = (1000,1,1))
    i = 0
    while i+4 < seql:
        seq[i,:,:] = 2
        seq[i+1,:,:] = np.random.choice([3, 4])
        seq[i+2,:,:] = np.random.choice([1, 5])
        i = i + 4
    return seq


def abstraction_illustration(seql = 1000):
    '''Shows that abstraction enables the model to learn better than chunks'''
    chunk = [1, 2, 3, 4, 5]
    abstraction = [[1, 2, 3], [4, 5], [2, 3, 4], [3, 4, 5], [2, 2], [4, 4], [5, 1]]
    seq = []
    while len(seq) <= 20000:
        seq = seq + chunk+abstraction[np.random.randint(len(abstraction))]+chunk
        seq.append(0)
    seq = seq[0:seql]
    seq = np.array(seq).reshape((len(seq),1,1))

    with open('sample_abstract_sequence.npy', 'wb') as f:
        np.save(f, seq)
    return seq


def exp2(control = False):
    if not control:
        training_seq = []
        testing_seq = []
        fixedparts = [4,5,6,5,4,6,6,5,4]
        for _ in range(0,40):
            seq = fixedparts.copy()
            seq.insert(2, np.random.choice([1,2,3]))
            seq.insert(6, np.random.choice([1,2,3]))
            seq.insert(10, np.random.choice([1,2,3]))
            training_seq.append(seq)

        testfixedparts = [5,6,4,4,6,5,6,4,5]
        for _ in range(0,24):
            seq = testfixedparts.copy()
            seq.insert(2, np.random.choice([1,2,3]))
            seq.insert(6, np.random.choice([1,2,3]))
            seq.insert(10, np.random.choice([1,2,3]))
            testing_seq.append(seq)

        return training_seq,testing_seq
    else:
        training_seq = []
        testing_seq = []
        fixedparts = [4, 5, 6, 5, 4, 6, 6, 5, 4]
        for _ in range(0, 40):
            seq = fixedparts.copy()
            seq.insert(2, 1)
            seq.insert(6, 2)
            seq.insert(10, 3)
            training_seq.append(seq)

        testfixedparts = [5, 6, 4, 4, 6, 5, 6, 4, 5]
        for _ in range(0, 24):
            seq = testfixedparts.copy()
            seq.insert(2, np.random.choice([1, 2, 3]))
            seq.insert(6, np.random.choice([1, 2, 3]))
            seq.insert(10, np.random.choice([1, 2, 3]))
            testing_seq.append(seq)

        return training_seq, testing_seq



def simple_seq():
    seq = np.array([[[0]], [[1]], [[2]], [[3]],[[4]],[[5]],[[6]],[[7]],[[8]],[[9]],[[10]],[[11]],[[12]]])
    return seq





def hierarchy1d():
    #================== Initialization Process ==================
    # level I
    # A = np.zeros([3, 1, 1])
    one = np.array([[[1]]])
    two = np.array([[[2]]])
    D = np.array([[[2]], [[1]]])
    C = np.array([[[1]], [[2]]])

    A = np.array([[[1]], [[2]], [[1]]])
    B = np.array([[[2]], [[1]], [[1]]])

    # level II
    CD = np.array([[[1]], [[2]], [[2]], [[1]]])#
    BD = np.array([[[2]], [[1]], [[1]], [[2]], [[1]]])#
    AB = np.array([[[1]], [[2]], [[1]], [[2]], [[1]], [[1]]]) #

    # level III
    ACD = np.array([[[1]], [[2]], [[1]], [[1]], [[2]], [[2]], [[1]]])#
    ABBD = np.array([[[1]], [[2]], [[1]], [[2]], [[1]], [[1]], [[2]], [[1]], [[1]], [[2]], [[1]]])
    E = np.zeros([1,1,1])
    E[0,0,0] = 0
    stim_set = [arr_to_tuple(E), arr_to_tuple(one), arr_to_tuple(two),
                arr_to_tuple(C), arr_to_tuple(D), arr_to_tuple(A), arr_to_tuple(B), arr_to_tuple(CD),
                arr_to_tuple(AB), arr_to_tuple(BD), arr_to_tuple(ACD), arr_to_tuple(ABBD)]

    '''Produce a generic generative model'''
    alpha = tuple([1 for i in range(0, len(stim_set))])  # coefficient for the flat dirichlet distribution
    probs = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    generative_marginals = {}
    # generative_marginals[arr_to_tuple(E)] = probs[0]
    for i in range(0, len(stim_set)):
        generative_marginals[stim_set[i]] = probs[i]
    cg.M = generative_marginals
    return cg



def generateseq(groupcond,seql = 600):
    seq = []
    if groupcond == 'c2':
        while len(seq) < seql:
            seq = seq + np.random.choice([[1,2],[3],[4]])
        return seq[0:seql]
    if groupcond == 'c3':
        while len(seq) < seql:
            seq = seq + np.random.choice([[1, 2, 3],[4]])
        return seq[0:seql]
    if groupcond == 'ind':
        while len(seq) < seql:
            seq = seq + [np.random.choice([1,2,3,4])]
        return seq[0:seql]