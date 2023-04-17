import numpy as np
import pandas as pd
from numpy.core._multiarray_umath import ndarray
from scipy import stats
from scipy.stats import chisquare
from math import log2
from chunks import *
from buffer import *
from test_Learning import *

def hcm_rational_v1(arayseq, cg):
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)
    seq, seql = convert_sequence(arayseq)  # loaded with the 0th observation
    seq = set(seq)
    seq = set([(0,0,0,1),(0,0,1,3),(0,2,2,2),(1,0,0,1),(1,0,1,3),(1,2,2,2),(2,0,0,1),(2,0,1,3),(2,2,2,2)])
    thischunk = Chunk([(0,0,0,1),(0,0,1,3),(0,2,2,2)], H=1, W=1)
    cont = list(thischunk.content)
    for obs in seq:
        inseq = False
        matching_elements = set()

        if obs[3] == cont[0][3]:
            t, x, y = obs[0] - cont[0][0], obs[1] - cont[0][1], obs[2] - cont[0][2]
            inseq = True
            matching_elements.add(obs)

            i = 1
            while i < len(cont) and inseq:
                point = cont[i]
                translated_chunk_point = (point[0] + t, point[1] + x, point[2] + y, point[3])
                if translated_chunk_point not in seq:
                    inseq = False
                else:
                    matching_elements.add(translated_chunk_point)
                i = i + 1

        if inseq:
            print('t = ', t)
            # seq.difference_update(matching_elements)
    return


def parse_sequence(cg, arayseq, seq, seql, candidate_set = set()):
    ''' candidate_set: a subset of chunks inside cg which is used to parse the sequence, only biggest chunk
    inside the candidate set can be identified '''

    t = 0
    Buffer = buffer(
        t, seq, seql, arayseq.shape[0]
    )  # seql: length of the current parsing buffer
    seq = seq  # reset the sequence
    cg.empty_counts()  # empty out all the counts before the next sequence parse
    maxchunksize = cg.getmaxchunksize()
    chunk_record = {}  # chunk ending time, and which chunk is it.
    seq_over = False
    i = 0# seqeunce items, for debugging proposes
    while seq_over == False:
        # identify latest ending chunks
        if len(candidate_set) < 1:
            candidate_set = set(cg.chunks.values())
        current_chunks, cg, dt, seq, chunk_record = identify_latest_chunks(
            cg, seq, chunk_record, Buffer.t, candidate_set
        )  # chunks
        seq = Buffer.refactor(seq, dt)
        i = i + 1

        cg = learning_and_update(
            current_chunks, chunk_record, cg, Buffer.t, threshold_chunk=False)
        maxchunksize = cg.getmaxchunksize()
        Buffer.reloadsize = maxchunksize + 1
        Buffer.checkreload(arayseq)
        seq_over = Buffer.checkseqover()
    print('in total, there are ', i , 'number of items being parsed')
    return cg, chunk_record


def hcm_rational(arayseq, cg, maxIter=2):
    """ returns chunking graph based on rational chunk learning
            Parameters:
                    arayseq(ndarray): Observational Sequences
                    cg (CG1): Chunking Graph

            Returns:
                    cg (CG1): Learned Representation from Data
    """
    # arayseq = arayseq[0:10,:,:]
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)
    seq, seql = convert_sequence(arayseq[0:1, :, :])  # loaded with the 0th observation
    Iter = 0
    independence = False
    while cg.independence_test() == False and Iter <= maxIter:
        print("============ empty out sequence ========== ")
        cg.empty_counts()
        print('test match   -----------------------  ')
        newtest = Test()
        newtest.test_adjacency_preadjacency_match(cg)
        print('end   -----------------------  ')

        cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks.items()))
        print('test match   -----------------------  ')
        newtest = Test()
        newtest.test_adjacency_preadjacency_match(cg)
        print('end   -----------------------  ')

        cg, cp = rational_learning(cg, n_update=10) # rationally learn until loss function do not converge
        #cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql)
        # print("Average Encoding Length is ER = ", cg.eval_avg_encoding_len())
        seq_over = False
        Iter = Iter + 1

    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))

    cg.abstraction_learning()

    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)

    cg, cp = rational_learning(cg, n_update=10)  # rationally learn until loss function do not converge

    cg.abstraction_learning()
    cg.empty_counts()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)
    cg, cp = rational_learning(cg)# learn chunks with variables

    cg.abstraction_learning()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)


    return cg


def meta_parse_sequence(chunk_record, cg, selected_chunks=[]):
    ''' based on chunk_record, search upward (from specific to concrete) in the representation tree to find consistent variables
    to explain the sequence.
        Input:
            selected_chunks: a subset of chunks used to parse the sequence, when empty, the shallowest var
        each parse gets the shallowest level of variables

        Can use a while loop to obtain all possible variable representations of the same sequence, the number of chunk
        record would be the depth of the representation graph.
        Each while loop would have a transition probability update, and a set of chunks to learn

        Output: a sequence, parsed via the variables that points to the concrete chunks in the chunkrecord
        '''
    abstract_chunk_record = chunk_record.copy()


    for t, cks in chunk_record.items():
        abstract_chunk_record[t] = []
        current_chunks = set()
        for ck,count in cks:# the second entry is frequency, can ignore for now.
            thischunk = cg.chunks[ck]
            if len(thischunk.variables)>0:# then pick which variable?
                var = np.random.choice(list(thischunk.variables))
                var.count = var.count + 1
                abstract_chunk_record[t].append((var.key, var.count))
            else:
                abstract_chunk_record[t].append((ck, count))

            current_chunks.add(ck)

        cg = learning_and_update(current_chunks, chunk_record, cg, t, threshold_chunk=True)

    return cg, abstract_chunk_record




def rational_learning(cg, n_update=10):
    """ given a learned representation, update chunks based on rank of joint occurrence frequency and hypothesis tests
            Parameters:
                n_update: the number of concatinations made based on the pre-existing cg records
            Returns:
                cg: chunking graph with empty chunks
    """
    candidancy_pairs = (
        []
    )  # iterate through all chunk pairs and find the chunks with the biggest candidancy


    for _previdx,_prevck in cg.chunks.items() | cg.variables.items(): # this iteration will be horribly slow.
        for _postidx in _prevck.adjacency:
            for _dt in _prevck.adjacency[_postidx].keys():
                _postck = cg.chunks[_postidx]
                _cat = combinechunks(_previdx, _postidx, _dt, cg)
                # hypothesis test
                if (
                    cg.hypothesis_test(_previdx, _postidx, _dt) == False
                ):  # reject null hypothesis
                    candidancy_pairs.append(
                        [
                            (_previdx, _postidx, _cat, _dt),
                            _prevck.adjacency[_postidx][_dt],
                        ]
                    )

    #
    # for _v in cg.variables.items(): # using _v as the previous chunk
    #     for _postidx in _v.adjacency:
    #         for _dt in _v.adjacency[_postidx].keys():
    #             _postck = cg.chunks[_postidx]
    #             _cat = combinechunks(_v, _postidx, _dt, cg)
    #             # hypothesis test
    #             if (
    #                 cg.hypothesis_test(_previdx, _postidx, _dt) == False
    #             ):  # reject null hypothesis
    #                 candidancy_pairs.append(
    #                     [
    #                         (_previdx, _postidx, _cat, _dt),
    #                         _prevck.adjacency[_postidx][_dt],
    #                     ]
    #                 )

    candidancy_pairs.sort(key=lambda tup: tup[1], reverse=True)
    print(candidancy_pairs)
    # number of chunk combinations allowed.
    for i in range(0, min(n_update,len(candidancy_pairs))):
        prev_idx, current_idx, cat, dt = candidancy_pairs[i][0]

        cg.chunking_reorganization(prev_idx, current_idx, cat, dt)
        newtest = Test()
        newtest.test_adjacency_preadjacency_match(cg)

        if i > len(candidancy_pairs):
            break
    return cg, candidancy_pairs


def learning_and_update(current_chunks, chunk_record, cg, t, threshold_chunk = False):
    '''
    Update transitions and marginals and decide to chunk
    t: finishing parsing at time t
    current_chunks_idx: the chunks ending at the current time point
    chunk_record: boundary record of when the previous chunks have ended.
    threshold_chunk: False when the function is used only for parsing and updating transition probabilities.
    
    TODO: Look backward and around to update transition according to chunk relations. '''
    n_t = 1# 2
    if len(chunk_record)>1:
        for chunk in current_chunks:
            d_t = 0 # the difference between the end of the current chunk and the end of the previous chunks
            temporal_length_chunk = cg.chunks[chunk].T
            while d_t <= temporal_length_chunk+n_t and len(chunk_record) > 1:# looking backward to find, padding length
                # adjacent chunks
                chunkendtime  = t - d_t
                if chunkendtime in list(chunk_record.keys()):
                    previous_chunks = chunk_record[chunkendtime]
                    # current_chunk_starting_time = t - temporal_length_of_current_chunk + 1  # the "current chunk" starts at:
                    for prev,_ in previous_chunks:
                        samechunk = prev == chunk
                        if not (samechunk and d_t==0):
                            if len(cg.chunks[prev].entailment) == 0:# do not check variable chunks
                                combined_chunk, dt = adjacency(prev, chunk, d_t, t, cg)
                                if combined_chunk is not None:
                                    # TODO: update variables associated with specific chunks as well
                                    # if the two chunks are adjacent to each other
                                    cg.chunks[prev].update_transition(cg.chunks[chunk], dt)
                                    if threshold_chunk: cg, chunked = threshold_chunking(prev, chunk, combined_chunk, dt, cg)
                d_t = d_t + 1
    return cg




def threshold_chunking(prev_key, current_key, combined_chunk, dt, cg):
    """combined_chunk: a new chunk instance
    dt: end_prev(inclusive) - start_post(exclusive)"""

    """cg: chunking graph
    learning function manipulates when do the chunk update happen
    chunks when some combination goes beyond a threshold"""
    chunked = False  # unless later it was verified to fit into the chunking criteria.
    cat = combined_chunk
    N = 5
    # the prevchunkindex needs to be in the list of all of the chunks
    prev = cg.chunks[prev_key]
    if dt in list(prev.adjacency.keys()):
        if current_key in list(prev.adjacency[dt].keys()):
            if prev.count > N:
                if prev.adjacency[dt][current_key] > N:
                    # strangely, this number affects the chunking behavior a lot.
                    if cg.hypothesis_test(prev_key, current_key,dt) == False:# reject null hypothesis
                        chunked = True
                        cg.chunking_reorganization(prev_key, current_key, cat, dt)

    return cg, chunked


import itertools


def check_overlap(array1, array2):
    output = np.empty((0, array1.shape[1]))
    for i0, i1 in itertools.product(np.arange(array1.shape[0]),
                                    np.arange(array2.shape[0])):
        if np.all(np.isclose(array1[i0], array2[i1])):
            output = np.concatenate((output, [array2[i1]]), axis=0)
    return output

def checkequal(chunk1,chunk2):
    if len(chunk1.content.intersection(chunk2.content)) == max(len(chunk1.content),len(chunk2.content)):
        return True
    else:
        return False

def evaluatesimilarity(chunk1, chunk2):
    return chunk1.checksimilarity(chunk2)

def adjacency(prev_key, post_key, time_diff, t, cg):
    # TODO: what if prev_idx and post_idx contains variables?
    # check adjacency should only check chunks with no variables, in other words, concrete chunks,
    # and their ancestors are tagged with this variable relationship

    # time_diff: difference between end of the post chunk and the end of the previous chunk (as in chunk record)
    ''' returns empty matrix if not chunkable '''
    # update transitions between chunks with a temporal proximity
    # chunk ends at the point of the end_point_chunk
    # candidate chunk ends at the point of the end_point_candidate_chunk
    if prev_key == post_key and time_diff == 0:# do not chunk a chunk by itself.
        return None, -100
    elif cg.chunks[prev_key].conflict(
        cg.chunks[post_key]
    ):  # chunks have conflicting content
        return None, -100
    else:
        prev = cg.chunks[prev_key]
        post = cg.chunks[post_key]
        e_post = t  # inclusive
        e_prev = t - time_diff  # inclusive
        s_prev = e_prev - prev.T  # exclusive
        s_post = e_post - post.T  # exclusive
        dt = e_prev - s_post
        delta_t = e_prev - max(
            s_post, s_prev
        )  # the overlapping temporal length between the two chunks
        # initiate a new chunk.
        prevcontent = prev.content.copy()
        postcontent = post.content.copy()

        if s_prev > s_post:
            # shift content in time:
            prev_shift = s_prev - s_post
            newprevcontent = set()
            for msrm in prevcontent:
                lmsrm = list(msrm)
                lmsrm[0] = lmsrm[0] + prev_shift  # pad the time dimension
                newprevcontent.add(tuple(lmsrm))
            prevcontent = newprevcontent

        prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W)
        prevchunk.T = prev.T
        prevchunk.H = prev.H
        prevchunk.W = prev.W

        if s_prev < s_post:
            post_shift = s_post - s_prev
            newpostcontent = set()
            for msrm in postcontent:
                lmsrm = list(msrm)
                lmsrm[0] = lmsrm[0] + post_shift
                newpostcontent.add(tuple(lmsrm))
            postcontent = newpostcontent

        postchunk = Chunk(list(postcontent), H=post.H, W=post.W)
        postchunk.T = post.T
        postchunk.H = post.H
        postchunk.W = post.W
        concat_chunk = prevchunk.concatinate(postchunk)
    return concat_chunk, dt
    #
    # if prev_key == post_key:
    #     return None, -100 # do not chunk a chunk by itself.
    # else:
    #     prev = cg.chunks[prev_key]
    #     post = cg.chunks[post_key]
    #     e_post = t  # inclusive
    #     e_prev = t - time_diff  # inclusive
    #     s_prev = e_prev - prev.T  # inclusive
    #     s_post = e_post - post.T  # inclusive
    #     if  s_post - s_prev > 0 and ~prev.check_adjacency_approximate(post,dt = s_post - s_prev):
    #         return None, -100
    #     elif s_post - s_prev < 0 and ~post.check_adjacency_approximate(prev, dt = s_prev - s_post):
    #         return None, -100
    #     else:
    #         dt = e_prev - s_post
    #         prevcontent = prev.ordered_content['0'].copy()
    #         postcontent = post.ordered_content['0'].copy()
    #
    #         if s_prev > s_post:
    #             # shift content in time:
    #             prev_shift = s_prev - s_post
    #             newprevcontent = set()
    #             for msrm in prevcontent:
    #                 lmsrm = list(msrm)
    #                 lmsrm[0] = lmsrm[0] + prev_shift  # pad the time dimension
    #                 newprevcontent.add(tuple(lmsrm))
    #             prevcontent = newprevcontent
    #
    #         prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W)
    #         prevchunk.T = prev.T
    #         prevchunk.H = prev.H
    #         prevchunk.W = prev.W
    #
    #         if s_prev<s_post:
    #             post_shift = s_post - s_prev
    #             newpostcontent = set()
    #             for msrm in postcontent:
    #                 lmsrm = list(msrm)
    #                 lmsrm[0] = lmsrm[0] + post_shift
    #                 newpostcontent.add(tuple(lmsrm))
    #             postcontent = newpostcontent
    #
    #         postchunk = Chunk(list(postcontent), H=post.H, W=post.W)
    #         postchunk.T = post.T
    #         postchunk.H = post.H
    #         postchunk.W = post.W
    #         concat_chunk = prevchunk.concatinate(postchunk, check = False)
    #     return concat_chunk, dt


def combinechunks(prev_key, post_key, dt, cg):
    # time_diff: difference between end of the post chunk and the end of the previous chunk
    ''' returns empty matrix if not chunkable '''
    # update transitions between chunks with a temporal proximity
    # chunk ends at the point of the end_point_chunk
    # candidate chunk ends at the point of the end_point_candidate_chunk
    if prev_key in cg.variables or post_key in cg.variables:
        Var = set()
        if prev_key in cg.variables:
            prev = cg.variables[prev_key]
            Var.add(prev)
        else:
            prev = cg.chunks[prev_key]

        if post_key in cg.variables:
            post = cg.variables[post_key]
            Var.add(post)
        else:
            post = cg.chunks[post_key]
        # combine chunks by merging ordered content
        combinedcontent = {}

        for dt, value in prev.ordered_content.items():
            combinedcontent[dt] = value

        dT = len(prev.ordered_content)
        for key, value in post.ordered_content.items():
            combinedcontent[str(int(key) + int(dT))] = value # shifted content representation

        combinedchunk = Chunk(list([]), variables=Var, ordered_content=combinedcontent)
        return combinedchunk

    else:
        post = cg.chunks[post_key]
        e_prev = 0
        l_t_post = post.T
        s_post = e_prev - dt  # start point is inclusive
        e_post = s_post + l_t_post
        if prev_key == post_key and e_post == e_prev:
            return None, -100 # do not chunk a chunk by itself.
        else:
            # TODO: double check if combine chunks and check adjacency generates the same chunk agglomeration
            prev = cg.chunks[prev_key]
            post = cg.chunks[post_key]
            e_prev = 0
            l_t_prev = prev.T
            l_t_post = post.T
            s_prev = e_prev - l_t_prev  # the exclusive temporal length
            s_post = e_prev - dt  # start point is inclusive
            e_post = s_post + l_t_post
            # dt = e_prev - s_post
            delta_t = e_prev - max(s_post, s_prev)# the overlapping temporal length between the two chunks
            t_chunk = max(e_post, e_prev) - min(e_post, e_prev) + delta_t + max(s_post, s_prev) - min(s_post, s_prev)# the stretching temporal length of the two chunks

            if t_chunk == l_t_prev and t_chunk == l_t_post and checkequal(prev, post):
                return None  # do not chunk a chunk by itself.
            else:
                # initiate a new chunk.
                prevcontent = prev.content.copy()
                postcontent = post.content.copy()
                if s_prev > s_post:  # post start first
                    # shift content in time:
                    prev_shift = s_prev - s_post
                    newprevcontent = set()
                    for msrm in prevcontent:
                        lmsrm = list(msrm)
                        lmsrm[0] = lmsrm[0] + prev_shift  # pad the time dimension
                        newprevcontent.add(tuple(lmsrm))
                prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W)
                prevchunk.T = prev.T
                prevchunk.H = prev.H
                prevchunk.W = prev.W

                if s_prev < s_post:  # prev start first
                    post_shift = s_post - s_prev
                    newpostcontent = set()
                    for msrm in postcontent:
                        lmsrm = list(msrm)
                        lmsrm[0] = lmsrm[0] + post_shift
                        newpostcontent.add(tuple(lmsrm))
                    postcontent = newpostcontent
                postchunk = Chunk(list(postcontent), H=post.H, W=post.W)
                postchunk.T = post.T
                postchunk.H = post.H
                postchunk.W = post.W
            return prevchunk.concatinate(postchunk)

            # if t_chunk == l_t_prev and t_chunk == l_t_post and checkequal(prev, post):
            #     return None# do not chunk a chunk by itself.
            # else:
            #     prevcontent = prev.content.copy()
            #     postcontent = post.content.copy()
            #     prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W)
            #     prevchunk.T = prev.T
            #     prevchunk.H = prev.H
            #     prevchunk.W = prev.W
            #     post_shift = dt
            #     newpostcontent = set()
            #     for msrm in postcontent:
            #         lmsrm = list(msrm)
            #         lmsrm[0] = lmsrm[0] + post_shift
            #         newpostcontent.add(tuple(lmsrm))
            #     postcontent = newpostcontent
            #     postchunk = Chunk(list(postcontent), H=post.H, W=post.W)
            #     postchunk.T = post.T
            #     postchunk.H = post.H
            #     postchunk.W = post.W
            # return prevchunk.concatinate(postchunk)

def add_singleton_chunk_to_M(seq, t, cg):
    '''Add the singleton chunk observed at the temporal point t of the seq to the set of chunks M'''
    H, W = seq.shape[1:]
    current_chunks = set()  # the set of chunks that end right before this moment t.
    for i in range(0, H):
        for j in range(0, W):
            chunk = np.zeros([1, H, W])
            chunk[0, i, j] = seq[t, i, j]
            if np.sum(np.abs(chunk)) > 0:  # not an empty chunks
                matchingchunk = cg.add_chunk_to_cg_class(chunk)
                current_chunks.add(matchingchunk)  # the chunk that ends right at the moment t
    return cg, current_chunks


def check_chunk_exist(chunk, M):
    for ck in M:
        if ck.profile == chunk:
            return True
    return False


def add_chunk_to_M(chunk, M):
    if chunk in list(M.keys()):
        M[chunk] = M[chunk] + 1
    else:
        M[chunk] = 1
    return M

def find_relevant_observations(seq, termination_time, t):

    H, W = seq.shape[1:]
    min_termin = min(termination_time.flatten())
    relevant_observations = np.zeros([int(t - min(termination_time.flatten())), H, W])
    for i in range(0, H):
        for j in range(0, W):
            relevant_t = (termination_time[i, j] - min_termin)
            relevant_observations[relevant_t:, i, j] = seq[termination_time[i, j] + 1:, i, j]
    relevant_observations, n_zero = refactor_observation(relevant_observations, return_max_k=True)  # cut such that
    # there is an event
    # happening
    # at the beginning.
    return relevant_observations, n_zero


# no chunks ends at this moment
def update_termination_time(current_chunks, chunk_termination_time, t):
    """For each spatial dimension, update the time that the latest chunk terminated
            :returns
                chunk_termination_time: shape: np.array(H,W), the height and width of the spatial dimension, in
                each dimension, the latest time that a chunk has ended. """
    for chunk in current_chunks:  # update chunk termination time for the current chunks
        chunk_termination_time[np.array(chunk)[-1, :, :] > 0] = t
    return chunk_termination_time


def identify_chunks(cg, seq, termination_time, t, previous_chunk_boundary_record):
    """Identify chunks to explain the current observation, and
    if the pre-existing chunk does not explain the current observation
    create new chunks
    # which one of them is predicted, which one of them is surprising.
    cg: chunking graph object
    seq: sequence of observation
    termination time: the time in each dimension where the last chunk has terminated.
    """
    "Starting from the location where the last chunk terminates, " \
    "needs explanation for what comes thereafter "
    # TODO: space for optimization to not parse the entire sequence
    relevant_observations, nzero = find_relevant_observations(seq, termination_time, t)
    # cut such that there is an event

    # happening at the beginning.
    if relevant_observations.shape[0] > 0:  # there are some observations to explain:
        # use the chunks to explain the current relevant observations.
        observation_to_explain = relevant_observations.copy()  # the sequence the same size as chunks:
        if len(list(cg.M.keys())) > 0:
            # check chunk match, start from the biggest chunk/the most likely
            # in the order of temporal slice, explain the observation one by one, until the entire observation is
            # cancelled
            # load things into current chunks
            current_chunks, termination_time, cg = identify_chunk_ending_with_M(observation_to_explain, cg,
                                                                                termination_time,t, previous_chunk_boundary_record)
            previous_chunk_boundary_record.append(current_chunks)
        else:  # nothing in the marginals, find the singleton chunks and load the marginals, relevant obsertvation size
            # = 1
            cg, current_chunks = add_singleton_chunk_to_M(seq, t, cg)
            delta_t = (tuple_to_arr(observation_to_explain).shape[0] - 1)
            t_end_best_matching_subchunk = t - delta_t
            termination_time = update_termination_time(current_chunks, termination_time, t_end_best_matching_subchunk)
            previous_chunk_boundary_record.append(current_chunks)
        # moved transition update to learning and update
        return current_chunks, cg, termination_time, previous_chunk_boundary_record
    else:
        return {}, cg, termination_time, previous_chunk_boundary_record


def pop_chunk_in_seq_full(chunkcontent, seqc, cg):#_c_
    for tup in chunkcontent: # remove chunk content one by one from the sequence
        seqc.remove(tup) # O(1)
    return seqc


def check_chunk_in_seq(chunk, seq): #_c_
    """ returns the size of match, how big the matching content is, and how big the chunk is, to be matched"""
    # returns the volume and the temporal length of the chunk content
    content = chunk.content
    if content.issubset(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0

def check_chunk_in_seq_boundary(chunk, seq):
    ''' Use chunk boundaries to identify chunks in the sequence
        Discrepencies of none-boundary contents are being ignored '''
    boundary = chunk.boundarycontent
    assert boundary is not None
    if boundary.issubset(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0

# compute the absolute difference between the observation and the template pixel

def check_chunk_in_seq_approximate(chunk, seq):  # more approximate
    '''TODO: Calculate the minimal approximate distance between chunk and sequence observations'''
    if chunk.check_match(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0

def pop_chunk_in_seq_boundary(chunk_idx, seqc, cg):
    # pop everything within the boundary of this chunk in seqc
    chunk = cg.chunks[chunk_idx]
    content = list(chunk.boundarycontent)
    T = sorted(content,key=lambda tup: tup[0])
    X = sorted(content,key=lambda tup: tup[1])
    Y = sorted(content,key=lambda tup: tup[2])

    tmax,tmin = T[-1][0], T[0][0]
    xmax, xmin = X[-1][1], X[0][1]
    ymax, ymin = Y[-1][2], Y[0][2]
    seqcc = seqc.copy()
    for tup in seqc:# remove chunk content one by one from the sequence
        if tup[0]>= tmin and tup[0]<=tmax and tup[1] >= xmin and tup[1]<=xmax and tup[2] >= ymin and tup[2]<=ymax:
            seqcc.remove(tup)
            if tup not in chunk.content:
                chunk.content.add(tup)# add sequential content within a chunk
    return seqcc


def timeshift(content, t):
    shiftedcontent = []
    for tup in list(content):
        lp = list(tup)
        lp[0] = lp[0] + t
        shiftedcontent.append(tuple(lp))
    return set(shiftedcontent)


def check_recursive_match(seqc, matchingcontent, chunk):
    '''when called, matching content is initialized to be set()
        seqc: a copy of the matching sequence
        '''
    assert(type(chunk) == Chunk),'matching type must be Chunk'
    match = False
    # TODO: iterate through chunk content to check consistency with concrete observation
    for ck in chunk.ordered_content:# need to add pre-existing components
        if type(ck) == set: # concrete chunks
            content = ck
            T = sorted(list(matchingcontent))[-1][0] + 1 if len(matchingcontent) > 0 else 0  # maximal finishing time
            temp = matchingcontent.copy()
            temp = temp.union(timeshift(content, T))
            if temp.issubset(seqc):
                match = True
                matchingcontent = temp
            else:
                return match, matchingcontent
        else: # variable
            for vck in ck.entailingchunks: # chunks inside a variable
                match, matchingcontent = check_recursive_match(seqc, matchingcontent, vck)
                if match: # no need to check other variables
                    break

    return match, matchingcontent



def identify_biggest_chunk(cg, seqc, candidate_set, checktype = 'full'):#_c_full
    # variables: the set of variable with their corresponding (associated chunk) to parse the sequence
    '''Chunk with a bigger size is priorized to explain the sequence'''
    matchingcontent = []
    t = 0
    if len(candidate_set) == 0:
        candidate_set = set(cg.chunks.values())

    def findmatch(current, seqc):
        # asset that at least one chunk in current explains seqc
        # if chunk contains no variable:
        # if chunk contains variables;
            # search for variables that match the sequences
        for chunk in current:  # what if there are multiple chunks that satisfies the relation?
            if len(chunk.variables) == 0:
                if chunk.ordered_content[0].issubset(seqc):
                    return chunk, chunk.ordered_content[0]
            else:
                match, matchingcontent = check_recursive_match(seqc, set(), chunk)
                if matchingcontent == None:
                    print('check')
                if match:
                    return chunk, matchingcontent
        return None, None

    if checktype == 'full':
        chunkidentification = check_chunk_in_seq
        pop_chunk_in_seq = pop_chunk_in_seq_full
    elif checktype == 'boundary':
        chunkidentification = check_chunk_in_seq_boundary
        pop_chunk_in_seq = pop_chunk_in_seq_boundary
    else:
        chunkidentification = check_chunk_in_seq
        pop_chunk_in_seq = pop_chunk_in_seq_full
    maxsizematch = 0
    maxchunk = None
    #current = set(cg.ancestors).intersection(candidate_set)# start parsing from the ancestor sets
    # current = set([a.content for a in cg.ancestors]).intersection(seqc)# start parsing from the ancestor sets
    current = cg.ancestors

    matching_chunks = set() # all chunks that are consistent with the sequence
    if len(cg.ancestors) == 0:
        maxchunk = None
    else:
        while len(current) > 0: # search until the bottom of the parsing tree from abstract to concrete
            c_star, c_star_content = findmatch(current, seqc)
            matching_chunks.add(c_star)
            if c_star is not None:
                c_star.parse = c_star.parse + 1 # TODO: make compatible
                current = c_star.cl
                maxchunk = c_star
            else:
                break


    # remove chunk from seq
    if maxchunk is None:
        maxchunk, cg = identify_singleton_chunk(cg, seqc)
        max_chunk_content = maxchunk.ordered_content[0]
        seqc = pop_chunk_in_seq(max_chunk_content, seqc, cg)  # pop identified chunks in sequence
    else:
        try:
            eligible_chunks = list(matching_chunks.intersection(candidate_set))
            maxchunk = eligible_chunks[-1]
        except(IndexError):
            print()
        max_chunk_content = maxchunk.ordered_content[0]
        seqc = pop_chunk_in_seq(max_chunk_content, seqc, cg)  # pop identified chunks in sequence

    return maxchunk, seqc # strange, maxchunk is indeed an instance inside cg, a chunk inside cg


def identify_singleton_chunk(cg, seqc):#_c_
    chunkcontent = [seqc[0]]
    chunk = Chunk(chunkcontent, H=cg.H, W=cg.W)
    cg.add_chunk(chunk, ancestor = True)
    return chunk, cg


def updatechunk(chunk, explainchunk, chunk_record, cg, t):
    if len(chunk.variables)== 0:
        explainchunk.append((chunk.key, chunk.T))
        cg.chunks[chunk.key].update()  # update chunk count among the currently identified chunks
        chunk_record = updatechunkrecord(chunk_record, chunk.key, int(chunk.T) + t-1, cg)
        return explainchunk, cg, chunk_record
    else:
        # TODO: need to adapt based on varibles potentially
        explainchunk.append((chunk.key, chunk.T))
        cg.chunks[chunk.key].update()  # update chunk count among the currently identified chunks
        chunk_record = updatechunkrecord(chunk_record, chunk.key, int(chunk.T) + t-1, cg)
        return explainchunk, cg, chunk_record

def identify_one_chunk(cg, seqc, explainchunk, chunk_record, t, candidate_set): #_c_
    max_chunk, seqc = identify_biggest_chunk(cg, seqc, candidate_set) # identify and pop chunks in sequence
    try:
        explainchunk, cg, chunk_record = updatechunk(max_chunk, explainchunk, chunk_record, cg,t)
    except(TypeError):
        print('Na Ni? ')
    # explainchunk.append((max_chunk_idx, int(cg.chunks[max_chunk_idx].T), list(cg.visible_chunk_list[max_chunk_idx])))
    # chunk_record = updatechunkrecord(chunk_record, max_chunk_idx, int(cg.chunks[max_chunk_idx].T) + t, cg)
    # cg.chunks[max_chunk_idx].update()  # update chunk count among the currently identified chunks
    return cg, seqc, chunk_record, explainchunk


def pop_chunk_in_seq_approximate(chunk_idx, seqc, cg):
    # pop everything within the boundary of this chunk in seqc
    chunk = cg.chunks[chunk_idx]
    print(' chunk content before ', chunk.content)
    print(' matching seq ', chunk.matching_seq)

    chunk.average_content() # average content
    print(' chunk content after ', chunk.content)

    cg.visible_chunk_list[chunk_idx] = chunk.content
    seqcc = seqc.copy()
    for m in list(chunk.matching_seq.keys()):
        for pt in chunk.matching_seq[m]:
            # can be the case that the same point in sequence matches multiple keys
            try:
                seqcc.remove(pt)
            except(ValueError):
                print(' one point matches two chunk points ')

    return seqcc


def identify_one_chunk_approximate(cg, seqc, explainchunk, chunk_record, t):
    maxsizematch = 0
    maxchunk = None
    for chunk in reversed(cg.chunks):# identify one chunk in sequence at one time...
        matchsize, match_len = check_chunk_in_seq_approximate(chunk, seqc)
        if matchsize > maxsizematch:
            maxsizematch = matchsize
            maxchunk = chunk.index

    # remove chunk from seq
    if maxchunk is None:
        maxchunk, cg = identify_singleton_chunk(cg, seqc)
        seqc = pop_chunk_in_seq_full(maxchunk, seqc, cg)
    else:
        seqc = pop_chunk_in_seq_approximate(maxchunk, seqc, cg)

    max_chunk_idx = maxchunk
    explainchunk.append((max_chunk_idx, int(cg.chunks[max_chunk_idx].T), list(cg.visible_chunk_list[max_chunk_idx])))
    chunk_record = updatechunkrecord(chunk_record, max_chunk_idx, int(cg.chunks[max_chunk_idx].T) + t, cg)
    cg.chunks[maxchunk.key].update()  # update chunk count among the currently identified chunks
    return cg, seqc, chunk_record, explainchunk

def check_seq_explained(seqc):#_c_
    # check whether there is a t = 0 in the seqc:
    if seqc == []: return True
    else: return seqc[0][0] != 0 # finished explaining the current time point


def updatechunkrecord(chunk_record, ckidx, endtime,cg, freq = True):
    if freq == True:
        if endtime not in list(chunk_record.keys()):
            chunk_record[endtime] = [(ckidx, cg.chunks[ckidx].count)]
        else:
            if ckidx not in chunk_record[endtime]: # one chunk can only be identified at one time point. when there are
                # multiple points that correspond to the same chunk, this chunk is identified as occurring once.
                chunk_record[endtime].append((ckidx, cg.chunks[ckidx].count))
    else:
        p = cg.chunks[ckidx].count / endtime
        if endtime not in list(chunk_record.keys()):
            chunk_record[endtime] = [(ckidx, p)]
        else:
            if ckidx not in chunk_record[endtime]: # one chunk can only be identified at one time point. when there are
                # multiple points that correspond to the same chunk, this chunk is identified as occurring once.
                chunk_record[endtime].append((ckidx, p))
    return chunk_record


def identify_latest_chunks(cg, seq, chunk_record, t, candidate_set):
    ''' use the biggest explainable chunk to parse the sequence and store chunks in the chunkrecord
        candidate_set: the specified set of variables '''
    def check_obs(s):  # there are observations at the current time point
        if s == []: return True # in the case of empty sequence
        else: return s[0][0] != 0 # nothing happens at the current time point
        # identify biggest chunks that finish at the earliest time point.
    t = t + 1# the current time (buffer updates after parsing)
    seq_explained = False
    seqc = seq.copy()
    explainchunk = []
    # find candidate chunks explaining the upcoming sequence, out of these candidate chunks, find the ones that
    # terminates at the earliest time point.
    no_observation = check_obs(seqc) # no observation in this particular time slice
    if no_observation:
        current_chunks_idx = []
        dt = int(1.0)
    else:
        while seq_explained == False:
            # find explanations for the upcoming sequence
            # record termination time of each biggest chunk used for explanation
            cg, seqc, chunk_record, explainchunk = identify_one_chunk(cg, seqc, explainchunk, chunk_record, t, candidate_set)
            seq_explained = check_seq_explained(seqc)
        # chunkrecord: {time: [chunkindex]} stores the finishing time (exclusive) of each chunk
        # decide which are current chunks so that time is appropriately updated
        explainchunk.sort(key=lambda tup: tup[1])# sort according to finishing time
        if len(seqc)>=1:
            dt = min(min(explainchunk, key=lambda tup: tup[1])[1], seqc[0][0])
        else:
            dt = min(explainchunk, key=lambda tup: tup[1])[1]
        current_chunks_idx = [item[0] for item in explainchunk if item[1] == dt] # move at the pace of the smallest identified chunk
        seq = seqc
    return current_chunks_idx, cg, dt, seq, chunk_record


def refactor(seq, current_chunk_idx, cg):
    # first, remove the identified current chunks from the sequence of observations
    if len(current_chunk_idx)>0:
        for idx,_ in current_chunk_idx:
            content = cg.chunks[idx].content
            for tup in content:# remove chunk content one by one from the sequence
                seq.remove(tup)

    seqcopy = []
    if seq !=[]:
        mintime = seq[0][0]
        for item in seq:
            listobs = list(item)
            listobs[0] = int(listobs[0] - mintime)
            # if listobs[0]>=0:
            seqcopy.append(tuple(listobs)) # there should not be negative ts, otherwise something is not explained properly
        return seqcopy
    else:
        return []


def check_subchunk_match(sequence_snapshot, chunk):
    """ check whether the subchunk of chunk matches with the sequence snapshot
        :returns True if there is a chunk match, false otherwise, when size is specified, the size of chunk match
        size: returns the size of the chunk, which is the number of observations that the chunk explains
        and also whether the sequence snapshot agrees with the chunk"""
    # whether the chunk could explain a subpart of the sequence snapshot
    elen = len(sequence_snapshot.intersection(chunk.content))
    return elen



def find_atomic_chunk(observation_to_explain):
    """:returns the first nonzero atomic unit"""

    H, W = observation_to_explain.shape[1:]
    for i in range(0, H):
        for j in range(0, W):
            chunk = np.zeros([1, H, W])
            if observation_to_explain[0, i, j] > 0:
                chunk[0, i, j] = observation_to_explain[0, i, j]  # group them by
                return chunk
    return np.zeros([1, H, W])


def rational_chunking(prev, current, combined_chunk, cg):
    """Updates chunking based on whether the new representation improves the evaluated loss function
        Updates the loss function associated with the new chunk, because the update of other chunks does not affect
        the loss function value estimation at the moment
        TODO: check the validity of this calculation method, possibly probability needs to be readjusted """
    w = 0.5 # tradeoff between accuracy and reaction time.
    cat = combined_chunk
    chunked = False
    P_s_giv_s_last = cg.get_transitional_p(prev, current)
    if P_s_giv_s_last is not None:
        Expected_reward_cat = P_s_giv_s_last * len(current) + (1 - P_s_giv_s_last) * (-1) * w * len(current)
        if Expected_reward_cat>0:
            chunked = True
            cg.chunking_reorganization(prev, current, cat)
    return cg, chunked


def arr_to_tuple(arr):
    test_list = arr.tolist()
    res = tuple([tuple([tuple(ele) for ele in sub]) for sub in test_list])
    return res


def tuple_to_arr(tup):
    return np.array(tup)


def update_current_chunks(current_chunks, M):
    # update the marginal frequency of the current chunk
    if len(current_chunk) > 0:
        for chunk in current_chunks:
            if arr_to_tuple(chunk) not in list(M.keys()):
                M[arr_to_tuple(chunk)] = 1.0
            else:
                M[arr_to_tuple(chunk)] = M[arr_to_tuple(chunk)] + 1.0
    return M


def refactor_observation(observation, return_max_k=False):
    ''' observation: matrix to refactor
    max_k: the steps of reduction does the observation reduces.'''
    max_k = 0
    sum_obs = np.sum(np.abs(observation), axis=(1, 2))

    # TODO: improve the code of refactorization
    if observation.shape[0] > 1:
        for k in range(1, observation.shape[0]):
            # if np.sum(np.abs(observation[0:k, :, :])) == 0:
            if np.sum(sum_obs[0:k]) == 0:
                if k > max_k:
                    max_k = k
    observation = observation[max_k:, :, :]
    if return_max_k:
        return observation, max_k
    else:
        return observation


def check_chunk_ending(chunk, observation):
    '''Given that a chunk could explain the observation, check on whether it is ending at the last column of the observation'''
    if len(chunk) == len(observation):
        return True
    else:
        return False


# convert frequency into probabilities
def transition_into_probabilities(chunk_f, chunk_pair_f, prev, current):
    # decide if the number of exposure is significant enough to be concatinated together as a chunk
    if (prev in list(chunk_f.keys()) and prev in list(
            chunk_pair_f.keys())) and current in list(
        chunk_pair_f[prev].keys()):
        sum_transition = 0
        for key in list(chunk_pair_f[prev].keys()):
            sum_transition += chunk_pair_f[prev][key]
        sum_marginals = 0
        for key in list(chunk_f.keys()):
            sum_marginals = sum_marginals + chunk_f[key]
        P_prev = chunk_f[prev] / sum_marginals
        P_current_giv_prev = chunk_pair_f[prev][current] / sum_transition
    else:
        P_prev = None
        P_current_giv_prev = None
    return P_prev, P_current_giv_prev


def measure_average_EUEP(chunk_f):
    avg_euep = 0
    for key in list(chunk_f.keys()):
        probs = chunk_f[key] / sum(list(chunk_f.values()))
        avg_euep = avg_euep + probs * len(list(key)) / np.log2(probs)
    return avg_euep


from time import time


def convert_sequence(seq):
    Observations = []
    T, H, W = seq.shape
    for t in range(0,T):
        for h in range(0,H):
            for w in range(0,W):
                v = seq[t,h,w]
                if v!=0:
                    Observations.append((t,h,w,seq[t,h,w]))
    return Observations, T

import time

def save_diagnostic_data(data, learningtime,cg,current_chunks_idx,maxchunksize):
    data['learning time'].append(learningtime)
    data['n_chunk'].append(len(cg.chunks))
    if len(current_chunks_idx) > 0:
        sizes = []
        for ck in cg.chunks:
            sz = ck.volume
            sizes.append(sz)
            if sz > maxchunksize: maxchunksize = sz
        data['chunk size'].append(maxchunksize)
    else:
        data['chunk size'].append(0)
    np.save('performance_data.npy', data)
    return data, maxchunksize


def save_chunk_record(chunkrecord, cg):
    import pickle
    ''' save chunk record for HCM learned on behaviorial data '''
    df = {}
    df['time'] = []
    df['chunksize'] = []
    for time in list(chunkrecord.keys()):
        df['time'].append(int(time))
        ckidx = chunkrecord[time][0][0]
        df['chunksize'].append(cg.chunks[ckidx].volume)
    with open('HCM_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)
    return

def hcm_learning(arayseq, cg, learn = True):
    '''Sequence is an n-d array
        Note that sequence should not be too long, otherwise subtraction would take a while for the algorithm
        when the learning handle is false, then only parsing, and no chunking will take place '''
    seql, H, W = arayseq.shape
    cg.update_hw(H,W)
    seq_over = False
    chunk_record = {} # chunk ending time, and which chunk is it.
    # dt: the time from the end of the previous chunk to the next chunk

    data = {}
    data['parsing time'] = []
    data['learning time'] = []
    data['n_chunk'] = []
    data['chunk size'] = []

    # seql: length of the current parsing buffer
    maxchunksize = 0
    if len(cg.chunks) > 0:
        sizes = []
        for ck in cg.chunks:
            sz = ck.volume
            sizes.append(sz)
            if sz > maxchunksize: maxchunksize = sz

    seq, seql = convert_sequence(arayseq[0:maxchunksize+1, :, :])# loaded with the 0th observation
    t = 0
    Buffer = buffer(t, seq, seql, arayseq.shape[0])


    while seq_over == False:
        currenttime = time.perf_counter()
        # identify latest ending chunks
        candidate_set = set(cg.chunks.values())
        print(cg, seq, chunk_record, Buffer.t, candidate_set)
        current_chunks_idx, cg, dt, seq, chunk_record = identify_latest_chunks(cg, seq, chunk_record, Buffer.t, candidate_set)  # chunks
        parsingtime = time.perf_counter() - currenttime
        data['parsing time'].append(parsingtime)
        currenttime = time.perf_counter()
        seq = Buffer.refactor(seq, dt)

        # that ends right before t. the last
        if len(current_chunks_idx)>0 and learn == True:
            cg = learning_and_update(current_chunks_idx, chunk_record, cg, Buffer.t)

        learningtime = time.perf_counter() - currenttime
        data, maxchunksize = save_diagnostic_data(data, learningtime, cg, current_chunks_idx, maxchunksize)
        # previous and current chunk
        if learn == True:
            cg.forget()
        # seq = Buffer.refactor(seq, current_chunks_idx, cg, dt)
        Buffer.reloadsize = maxchunksize + 1
        Buffer.checkreload(arayseq)
        seq_over = Buffer.checkseqover()
    return cg, chunk_record

def reload(seq, seql, arraysequence,t, max_chunksize):
    # reload arraysequence starting from time point t to the set sequence representations
    # t: the current time point
    T = seql # current buffer length
    time = t + T
    relevantarray = arraysequence[time:time+max_chunksize, :, :]
    _, H, W = arraysequence.shape
    for tt in range(0, min(max_chunksize,relevantarray.shape[0])):
        for h in range(0,H):
            for w in range(0,W):
                v = relevantarray[tt,h,w]
                seq.append((T + tt, h, w, v))
    seql = seql + min(max_chunksize,relevantarray.shape[0])
    return seq, seql



def independence_test(pM, pT,N):
    # f_obs, f_exp = None, ddof = 0
    f_obs = []
    f_exp = []
    B = list(pM.keys())
    for cl in B:
        for cr in B:
            pclcr = pM[cl]*pM[cr]*N # expected number of observations
            oclcr = pM[cl]*pT[cl][cr]*N # number of observations
            f_exp.append(pclcr)
            f_obs.append(oclcr)
    df = (len(B) - 1)**2
    _,pvalue = stats.chisquare(f_obs, f_exp=f_exp, ddof=df)

    if pvalue < 0.05:
        return False # reject independence hypothesis, there is a correlation
    else:
        return True


def get_minimal_complete_cks(seq, cg):

    for i in range(0, seq.shape[0]):
        atomic_chunk = np.zeros([1, 1, 1])
        atomic_chunk[0,0,0] = seq[i, 0, 0]
        if arr_to_tuple(atomic_chunk) not in list(cg.M.keys()):
            cg.M[arr_to_tuple(atomic_chunk)] = 1
            cg.T[arr_to_tuple(atomic_chunk)] = {}
            cg.add_chunk_to_vertex(arr_to_tuple(atomic_chunk))
        else:
            cg.M[arr_to_tuple(atomic_chunk)] = cg.M[arr_to_tuple(atomic_chunk)] + 1

    return cg


def rational_chunking_all_info(seq, cg, maxit = 10):
    """" Rational chunk learner with the access to information about all the sequence
    Implemented in the hierarchical spatial temporal sequence
    """
    def hypothesis_test(cl,cr,pM,pT,N):
        p1p1 = pM[cl]*pM[cr]
        p1p0 = pM[cl]*(1 - pM[cr])
        p0p1 = (1 - pM[cl])*pM[cr]
        p0p0 = (1 - pM[cl])*(1 - pM[cr])

        op1p1 = pM[cl]*pT[cl][cr]
        op1p0 = pM[cl]*(1 - pT[cl][cr])
        op0p1 = 0
        op0p0 = 0

        for ncl in list(pT.keys()):
            if ncl != cl:
                op0p1 = op0p1 + pT[ncl][cr]*pM[ncl]
                op0p0 = op0p0 + (1-pT[ncl][cr])*pM[ncl]
        # if any cell contains less than 5 observations, need to
        if op0p0*N <=5 or op1p1*N <=5 or op1p0*N <=5 or op0p1*N <=5:
            return True# cannot continute the test because of lack of sample size

        chisquare = N*(p1p1*((op1p1 - p1p1)/p1p1)**2 + p1p0*((op1p0 - p1p0)/p1p0)**2 + p0p1*((op0p1 - p0p1)/p0p1)**2
                       + p0p0*((op0p0 - p0p0)/p0p0)**2)
        pvalue = stats.chi2.pdf(chisquare, 1)
        if pvalue < 0.0005:
            return False # reject independence hypothesis, there is a correlation
        else:
            return True

    # get minimally complete chunk sets
    if len(cg.M) == 0:
        cg = get_minimal_complete_cks(seq, cg)

    if len(cg.M)>0:
        pM, pT, _,N = partition_seq_hastily(seq, list(cg.M.keys()))

    proposed_chunk_p = {}
    for cl in list(pT.keys()):
        p_cl = pM[cl]
        for cr in list(pT[cl].keys()):
            cr_giv_cl = pT[cl][cr]
            p = p_cl * cr_giv_cl
            # check independence:
            if hypothesis_test(cl,cr,pM,pT,N) == False: # there is a correlation
                if np.sum(tuple_to_arr(cl))>0 and np.sum(tuple_to_arr(cr))>0:
                    clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                    # TODO: proposed chunk should rank with its parents.
                    proposed_chunk_p[clcr] = (cl, cr, p)
    sorted_proposal = {k: v for k, v in sorted(proposed_chunk_p.items(), key=lambda item: item[1][2])}

    # find the maximum transition probability, try chunking it
    it = 0  # number of iteration
    while independence_test(pM, pT,N) == False and len(list(sorted_proposal.keys())) > 0:
        #''alternatively, while there are still dependencies between the chunks, keep searching for representation \n'
        # '        until they become independent')
        print(sorted_proposal.keys())
        new_chunk = list(sorted_proposal.keys())[-1]
        cl = sorted_proposal[new_chunk][0]
        cr = sorted_proposal[new_chunk][1]

        bag_of_chunks = list(pM.keys())
        while new_chunk in bag_of_chunks:
            print('the new chunk is already existing in the bag of chunks')
            sorted_proposal.pop(new_chunk)
            proposed_chunk_p.pop(new_chunk)
            new_chunk = list(sorted_proposal.keys())[-1]
            cl = sorted_proposal[new_chunk][0]
            cr = sorted_proposal[new_chunk][1]
        print(sorted_proposal)
        print(new_chunk)

        cg.add_chunk_to_vertex(new_chunk, left= cl, right = cr)
        new_bag_of_chunks = bag_of_chunks.copy()
        new_bag_of_chunks.append(new_chunk)
        new_pM, new_pT,_,N = partition_seq_hastily(seq, new_bag_of_chunks)
        pM = new_pM
        pT = new_pT
        cg.M = new_pM
        cg.T = new_pT

        proposed_chunk_p = {}
        for cl in list(new_pM.keys()):
            p_cl = new_pM[cl]
            for cr in list(new_pT[cl].keys()):
                cr_giv_cl = new_pT[cl][cr]
                p = p_cl * cr_giv_cl
                if hypothesis_test(cl, cr, pM, pT, N) == False:  # there is a correlation
                    if np.sum(tuple_to_arr(cl)) > 0 and np.sum(tuple_to_arr(cr)) > 0:
                        clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                        proposed_chunk_p[clcr] = (cl, cr, p)
        sorted_proposal = {k: v for k, v in sorted(proposed_chunk_p.items(), key=lambda item: item[1][2])}

        it = it + 1
        if it > maxit or len(list(sorted_proposal.keys())) == 0:# in case there is no proposal, exit.
            break

    return cg



# helper functions
def eval_pM_pT(bag_of_chunks,partitioned_sequence, freq = True):

    """checked"""
    M = {}
    T = {}
    for chunk in bag_of_chunks:
        parsed_chunk = chunk
        if freq == True:
            M[chunk] = get_pM_from_partitioned_sequence(parsed_chunk, partitioned_sequence, freq=True)
        else:
            M[chunk] = get_pM_from_partitioned_sequence(parsed_chunk, partitioned_sequence, freq=False)
    for cl in bag_of_chunks:
        T[cl] = {}
        for cr in bag_of_chunks:
            T[cl][cr] = get_pT_from_partitioned_sequence(cl, cr, partitioned_sequence, freq=freq)
    return M, T


def get_pT_from_partitioned_sequence(cl, cr, partitioned_sequence, freq = True):
    '''checked'''
    '''Get the estimated empirical probability of P(chunk2|chunk1),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    chunk1_chunk2 = 0
    # the transition probability from chunk1 to chunk2
    # get P(chunk2|chunk1)
    not_over = True
    i = 0
    for candidate_chunk in partitioned_sequence:
        if candidate_chunk == cl:
            chunk1_count += 1
            if i + 1 < len(partitioned_sequence):
                candidate2 = partitioned_sequence[i + 1]
                if candidate2 == cr:
                    chunk1_chunk2 += 1
        i = i + 1
    if freq:
        return chunk1_count
    else:
        if chunk1_count > 0:
            return chunk1_chunk2 / chunk1_count
        else:
            return 0.0  # returns 0 if there is no occurrance for the first probabilility


def get_pM_from_partitioned_sequence(chunk, partitioned_sequence, freq = True):
    '''checked'''

    '''Get the estimated empirical probability of P(chunk),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    for candidate_chunk in partitioned_sequence:
        if candidate_chunk == chunk:
            chunk1_count += 1
    if freq:
        return chunk1_count
    else:
        if chunk1_count > 0:
            return chunk1_count / len(partitioned_sequence)
        else:
            return 0.0








