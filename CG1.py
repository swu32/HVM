from Learning import *
from chunks import *
import numpy as np
import copy

class CG1:
    """

    ...

    Attributes
    ----------
    vertex_list : list
        chunk objects learned
    vertex_location : list
        graph location of the corresponding chunk
    edge_list : list
        Edge information about which chunk combined with which in the model

    Methods
    -------

    """
    # TODO: record which chunk created when
    # TODO: record on the dependencies between the chunks

    def __init__(self, y0=0, x_max=0, DT = 0.01, theta=0.75):
        """DT: deletion threshold"""
        # vertex_list: list of vertex with the order of addition
        # each item in vertex list has a corresponding vertex location.
        # edge list: list of vertex tuples
        self.vertex_list = [] # list of the chunks
        # the concrete and abstract chunk list together i
        self.y0 = y0 # the initial height of the graph, used for plotting
        self.x_max = x_max # initial x location of the graph
        self.chunks = {}# a dictonary with chunk keys and chunk tuples
        self.concrete_chunks = []# no entailment
        self.ancestors = []# list of chunks without parents
        self.theta = theta# forgetting rate
        self.deletion_threshold = DT
        self.H = 1 # default
        self.W = 1
        self.zero = None
        self.relational_graph = False


    def get_N(self):
        """returns the number of parsed observations"""
        assert len(self.chunks)>0
        N = 0
        for i in self.chunks:
            N = N + self.chunks[i].count
        return N

    def get_N_transition(self, dt = None):
        """returns the number of parsed observations"""
        assert len(self.chunks)>0

        if dt == None:
            N_transition = 0
            for chunk in self.chunks:
                for dt in self.chunks[chunk].adjacency:
                    N_transition = N_transition + sum(self.chunks[chunk].adjacency[dt].values())

            return N_transition
        else:
            N_transition = 0
            for chunk in self.chunks:
                if dt in self.chunks[chunk].adjacency:
                    N_transition = N_transition + self.chunks[chunk].adjacency[dt].values()
            return N_transition


    def empty_counts(self):
        """empty count entries and transition entries in each chunk"""
        for ck in self.chunks:
            ck.count = 0
            ck.empty_counts()
        return


    def hypothesis_test(self,clidx,cridx,dt):
        cl = self.chunks[clidx]
        cr = self.chunks[cridx]
        assert len(cl.adjacency)>0
        assert dt in list(cl.adjacency.keys())
        N = self.get_N()
        N_transition = self.get_N_transition(dt = dt)

        N_min = 5
        # Expected
        ep1p1 = (cl.count/N) * (cr.count/N) * N_transition
        ep1p0 = (cl.count/N) * (N - cr.count)/N * N_transition
        ep0p1 = (N - cl.count)/N * (cr.count/N) * N_transition
        ep0p0 = (N - cl.count)/N * (N - cr.count)/N * N_transition

        # Observed
        op1p1 = cl.adjacency[dt][cridx]
        op1p0 = cl.get_N_transition(dt) - cl.adjacency[dt][cridx]
        op0p1 = 0
        op0p0 = 0
        for ncl in list(self.chunks):# iterate over p0, which is the cases where cl is not observed
            if ncl != cl:
                if dt in list(ncl.adjacency.keys()):
                    if cridx in list(ncl.adjacency[dt].keys()):
                        op0p1 = op0p1 + ncl.adjacency[dt][cridx]
                        for ncridx in list(ncl.adjacency[dt].keys()):
                            if ncridx != cr:
                                op0p0 = op0p0 + ncl.adjacency[dt][ncridx]

        if op0p0 <= N_min or op1p0 <= N_min or op1p1 <= N_min or op0p1 <= N_min:
            return True
        else:
            _, pvalue = stats.chisquare([op1p1,op1p0,op0p1,op0p0], f_exp=[ep1p1,ep1p0,ep0p1,ep0p0], ddof=1)
            # print('p value is ', pvalue)
            if pvalue < 0.05:
                return False # reject independence hypothesis, there is a correlation
            else:
                return True


    def getmaxchunksize(
        self,
    ):  # TODO: alternatively, update this value upon every chunk creation
        maxchunksize = 0
        if len(self.chunks) > 0:
            for ck in self.chunks:
                if ck.volume > maxchunksize:
                    maxchunksize = ck.volume

        return maxchunksize

    def observation_to_tuple(self,relevant_observations):
        """relevant_observations: array like object"""
        index_t, index_i, index_j = np.nonzero(relevant_observations)# observation indexes
        value = [relevant_observations[t,i,j] for t,i,j in zip(index_t,index_i,index_j) if relevant_observations[t,i,j]>0]
        content = set(zip(index_t, index_i, index_j, value))
        maxT = max(index_t)
        return (content,maxT)

    def update_hw(self, H, W):
        self.H = H
        self.W = W
        return


    def get_M(self):
        return self.M

    def get_nonzeroM(self):
        nzm = list(self.M.keys()).copy()
        nzmm = nzm.copy()
        nzmm.remove(self.zero)

        return nzmm

    def reinitialize(self):
        # use in the case of reparsing an old sequence using the learned chunks.
        for ck in self.chunks:
            ck.count = 0
        return

    def graph_pruning(self):
        # prune representation graph
        init = self.ancestors.copy()

        for ck in init:
            this_chunk = ck
            while len(this_chunk.cl) > 0: # has children

                if len(this_chunk.cl) == 1: #only one children
                    if this_chunk.acl == []:# ancestor node
                        self.ancestors.pop(this_chunk)
                        self.ancestors.__add__(this_chunk.cl)
                    else:
                        ancestor = this_chunk.acl
                        ancestor.cl.add(this_chunk.cl)
                        this_chunk.acr.cr.pop(this_chunk)
                        this_chunk.acr = []


                    for rightkid in this_chunk.cr:
                        this_chunk.cl.__add__(rightkid)
                        rightkid.cl = ancestor# TODO: can add right chunk ancestor to children as well.
        return



    def get_T(self):
        # TODO: have not specified whether the transition is across space, or time, or space time, this would
        #  make it difficult to generate the chunk. IDEA: use delta t to encode the temporal transition across time.
        return self.T

    def get_chunk_transition(self, chunk):
        if chunk in self.T:
            return chunk.transition
        else:
            print(" no such chunk in graph ")
            return

    def convert_chunks_in_arrays(self):
        '''convert chunk representation to arrays'''
        for chunk in self.chunks:
            chunk.to_array()
        return

    def save_graph(self, name = '', path = ''):
        import json
        '''save graph configuration for visualization'''
        chunklist = []
        for ck in self.chunks:
            ck.to_array()
            chunklist.append(ck.arraycontent)
        data = {}
        data['vertex_location'] = self.vertex_location
        data['edge_list'] = self.edge_list
        # chunklist and graph structure is stored separately
        Name = path + name + 'graphstructure.json'
        a_file = open(Name, "w")
        json.dump(data, a_file)
        a_file.close()
        # np.save('graphchunk.npy', chunklist, allow_pickle=True)
        return

    def check_and_add_to_dict(self, dictionary, key):
        if key in dictionary:
            dictionary[key] = dictionary[key] + 1
        else:
            dictionary[key] = 1
        return dictionary

    # update graph configuration
    def add_chunk(self, newc, leftkey= None, rightkey = None):
        # TODO: add time when the chunk is being created
        self.vertex_list.append(newc.key)
        self.chunks[newc.key] = newc # add observation

        newc.index = self.chunks.index(newc)
        newc.H = self.H # initialize height and weight in chunk
        newc.W = self.W
        # compute the x and y location of the chunk based on pre-existing
        # graph configuration, when this chunk first emerges
        if leftkey is None and rightkey is None:
            x_new_c = self.x_max + 1
            y_new_c = self.y0
            self.x_max = x_new_c
            newc.vertex_location = [x_new_c, y_new_c]
        else:
            leftparent = self.chunks[leftkey]
            rightparent = self.chunks[rightkey]
            l_x, l_y = leftparent.vertex_location
            r_x, r_y = rightparent.vertex_location[rightkey]
            x_c = (l_x + r_x)*0.5
            y_c = self.y0
            self.vertex_location = [x_c, y_c]
            self.y0 = self.y0 + 1

            leftparent.cl = self.check_and_add_to_dict(leftparent.cl, newc)
            rightparent.cr = self.check_and_add_to_dict(rightparent.cr, newc)
            newc.acl = self.check_and_add_to_dict(newc.acl, leftparent)
            newc.acr = self.check_and_add_to_dict(newc.acl, rightparent)

        return

    def relational_graph_refactorization(self, newc):
        # find biggest common intersections between newc and all other previous chunks in the set
        for chunk in self.vertex_list:
            if chunk.children == []:# start from the leaf node to find the biggest smaller intersection.
                max_intersect = newc.content.intersection(chunk.content)
                if max_intersect in self.visible_chunk_list: # create an edge between the biggest smaller intersection and newc
                    idx_max_intersect = self.visible_chunk_list[max_intersect].idx
                    if ~self.check_ancestry(chunk, max_intersect):# link max intersect to newc
                        self.edge_list.append((idx_max_intersect, self.chunks[newc].idx))
                        self.chunks[idx_max_intersect].children.append(newc)
                    else: # in chunk's ancestors:
                        print('intersection already exist')
                        self.edge_list.append((idx_max_intersect, self.chunks[newc].idx))
                        chunk.children.append(newc)
                else: #if not, add link from this chunk to newc and some chunk
                    max_intersect_chunk = Chunk(list(max_intersect), H= chunk.H, W= chunk.W)
                    self.add_chunk(max_intersect_chunk, leftidx=None, rightidx=None)
                    self.edge_list.append((self.chunks[max_intersect_chunk].idx, self.chunks[newc].idx))
                    self.edge_list.append((self.chunks[max_intersect_chunk].idx, self.chunks[chunk].idx))
                    max_intersect_chunk.children.append(newc)
                    max_intersect_chunk.children.append(chunk)
        return
    #
    #
    # def relational_graph_refactorization(self, newc):
    #     # find biggest smaller intersections
    #     # smallest bigger intersections
    #     biggest_smaller_intersections = set()
    #     smallest_bigger_intersections = set()
    #
    #     smaller_intersections = {}
    #     bigger_intersections = {}
    #
    #     bsi = 0
    #     sbi = 1000
    #     for chunk in self.vertex_list:
    #         max_intersect = newc.content.intersection(chunk.content)
    #         if len(max_intersect) == len(newc.content): # newc is an intersection
    #             ? Is chunk the smallest bigger intersection?
    #             E(newc, chunk)
    #         elif len(max_intersect) == len(chunk.content):  # chunk is an intersection
    #             ? Is chunk the biggest smaller intersection?
    #         else:
    #
    #
    #
    #         if len(max_intersect) in smaller_intersections:
    #         if chunk.content in newc.content: # biggest smaller intersection
    #         if newc.content in chunk.content:# smallest bigger intersection
    #
    #     return


    # def variable_identification():
    #     # identify tree branch structure and calculate the gain of merging



    def relational_graph_refactorization(self, newc):
        # find variable amongst chunks
        return

    def check_ancestry(self,chunk,content):
        # check if content belongs to ancestor
        if chunk.parents == []:return content!=chunk.content
        else:return np.any([self.check_ancestry(parent,content) for parent in chunk.parents])

    def update_empty(self, n_empty):
        """chunk: nparray converted to tuple format
        Every time when a new chunk is identified, this function should be called """
        ZERO = self.zero
        self.M[ZERO] = self.M[ZERO] + n_empty
        return



    def check_chunk_in_M(self,chunk):
        """chunk object"""
        # content should be the same set
        for otherchunk in M:
            intersect = otherchunk.intersection(chunk)
            if len(intersect) ==chunk.volume:
                return otherchunk
        return


    def check_chunkcontent_in_M(self,chunkcontent):
        if chunkcontent in self.chunks:
            return self.chunks[chunkcontent]
        else:
            return None

    def add_chunk_to_cg_class(self, chunkcontent):
        """chunk: nparray converted to tuple format
        Every time when a new chunk is identified, this function should be called """
        matchingchunk = self.check_chunkcontent_in_M(chunkcontent)
        if matchingchunk!= None:
            self.M[matchingchunk] = self.M[matchingchunk] + 1
        else:
            matchingchunk = Chunk(chunkcontent, H=self.H, W=self.W) # create an entirely new chunk
            self.M[matchingchunk] = 1
            self.add_chunk(matchingchunk)
        return matchingchunk

    # convert frequency into probabilities

    def forget(self):
        """ discounting past observations if the number of frequencies is beyond deletion threshold"""
        for chunk in self.chunks:
            chunk.count = chunk.count* self.theta
            # if chunk.count < self.deletion_threshold: # for now, cancel deletion threshold, as index to chunk is
            # still vital
            #     self.chunks.pop(chunk)
            for dt in list(chunk.adjacency.keys()):
                for adj in list(chunk.adjacency[dt].keys()):
                    chunk.adjacency[dt][adj] = chunk.adjacency[dt][adj]* self.theta
                    if chunk.adjacency[dt][adj]<self.deletion_threshold:
                        chunk.adjacency[dt].pop(adj)

        return


    def checkcontentoverlap(self, content):
        '''check of the content is already contained in one of the chunks'''
        if content in self.chunks:
            return self.chunks[content]
        else:
            return None

    def chunking_reorganization(self, prevkey, currentkey, cat, dt):
        ''' Reorganize marginal and transitional probability matrix when a new chunk is created by concatinating prev and current '''
        prev = self.chunks[prevkey]
        current = self.chunks[currentkey]
        """Model hasn't seen this chunk before:"""
        chunk = self.checkcontentoverlap(cat.key)
        if chunk is None:
            # add concatinated chunk to the network
            # TODO： add chunk to vertex
            self.add_chunk(cat, leftkey=prevkey, rightkey=currentkey)
            # iterate through all chunk transitions that could lead to the same concatination chunk
            cat.count = prev.adjacency[dt][currentkey] # need to add estimates of how frequent the joint frequency occurred
            prev.count = prev.count - cat.count # reduce one sample observation from the previous chunk
            current.count = current.count - cat.count
            prev.adjacency[current][dt] = 0

            cat.adjacency = copy.deepcopy(current.adjacency)
            # check if there are other pathways that arrive at the same chunk
            ck = cat
            while len(ck.acl) > 0:
                ck = np.random.choice(ck.acl.keys())

            while len(ck.cl) > 0:
                for _cr in ck.adjacency:
                    for _dt in ck[_cr]:
                        if _cr != currentkey and ck.key != prevkey and _dt != dt:
                            _cat = combinechunks(ck.key, _cr, _dt, self)
                            if _cat!=None:
                                if _cat==cat:
                                    # TODO: may need to merge nested dictionary
                                    _cat_count =  self.chunks[ck].adjacency[_cr][_dt]
                                    cat.count = cat.count + _cat_count
                                    ck.count = ck.count - _cat_count
                                    _cr.count = _cr.count - _cat_count
                                    ck.adjacency[_cr][_dt] = 0

                ck = ck.cl
        else:
            chunk.count = chunk.count + prev.adjacency[dt][currentkey]  # need to add estimates of how frequent the joint frequency occurred
            prev.count = prev.count - cat.count# reduce one sample observation from the previous chunk
            current.count = current.count - cat.count
            prev.adjacency[current][dt] = 0

        return

    def evaluate_merging_gain(self, intersect, intersect_chunks):
        return

    def set_variable_adjacency(self, variable, entailingchunks):
        transition = {}
        for idx in entailingchunks:
            ck = self.chunks[idx]
            ck.abstraction.add(self)
            for _dt in list(ck.adjacency.keys()):
                if _dt not in list(transition.keys()):
                    transition[_dt] = ck.adjacency[_dt]
                else:
                    transition[_dt] = transition[_dt] + ck.adjacency[_dt]
        variable.adjacency = transition
        return

    def variable_finding(self, cat):
        v = 3 # threshold of volume of intersection
        app_t = 3# applicability threshold
        '''cat: new chunk which just entered into the system
        find the intersection of cat with the pre-existing chunks '''
        # (content of intersection, their associated chunks) ranked by the applicability threshold
        # alternatively, the most applicable intersection:
        max_intersect = None
        max_intersect_count = 0
        max_intersect_chunks = [] # chunks that needs to be merged
        for ck in self.chunks:
            intersect = ck.content.intersection(cat.content)
            intersect_chunks = []
            c = 0  # how often this intersection is applicable across chunks
            if len(intersect) != len(cat.content) and len(intersect) > v:# not the same chunk
                # look for overlap between this intersection and other chunks:
                for ck_ in self.chunks:# how applicable is this intersection, to other previously learned chunks
                    if ck_.content.intersection(intersect) == len(intersect):
                        c = c + 1
                        intersect_chunks.append(ck_)
            if c > max_intersect_count and c >= app_t:
                # atm. select intersect with the max intersect count
                # TODO: can be ranked based on merging gain
                max_intersect_count = c
                max_intersect_chunks = intersect_chunks
                max_intersect = intersect
        if max_intersect!=None: # reorganize chunk list to integrate with variables
            self.merge_chunks(max_intersect, max_intersect_chunks, max_intersect_count)
        return

    def merge_chunks(self, max_intersect, max_intersect_chunks, max_intersect_count):
        # create a new chunk with intergrated variables.
        for ck in max_intersect_chunks:
            ck.content = ck.content - max_intersect
        var = Variable(max_intersect_chunks, totalcount=max_intersect_count)
        self.set_variable_adjacency(var, max_intersect_chunks)

        chk = None # find if intersection chunk exists in the past
        for ck in self.chunks:
            if ck.content.intersection(max_intersect) == len(ck.content):
                chk = ck
        if chk == None: #TODO: add new chunk here
            chk = Chunk(max_intersect, count=max_intersect_count)
        else:
            chk.count = max_intersect_count

        # TODO: add new variable chunk here.
        chk_var = Chunk([chk, var])# an agglomeration of chunk with variable is created


        return

    def pop_transition_matrix(self, element):
        """transition_matrix:
        delete the entries where element follows some other entries.
        """
        transition_matrix = self.T
        # pop an element out of a transition matrix
        if transition_matrix != {}:
            # element should be a tuple
            if element in list(transition_matrix.keys()):
                transition_matrix.pop(element)
                # print("pop ", item, 'in transition matrix because it is not used very often')
            for key in list(transition_matrix.keys()):
                if element in list(transition_matrix[
                                       key].keys()):  # also delete entry in transition matrix
                    transition_matrix[key].pop(element)
        return

    def print_size(self):
        return len(self.vertex_list)

    def sample_from_distribution(self, states, prob):
        """
        states: a list of chunks
        prob: another list that contains the probability"""
        prob = [k / sum(prob) for k in prob]
        cdf = [0.0]
        for s in range(0, len(states)):
            cdf.append(cdf[s] + prob[s])
        k = np.random.rand()
        for i in range(1, len(states) + 1):
            if (k >= cdf[i - 1]):
                if (k < cdf[i]):
                    return states[i - 1], prob[i - 1]

    def sample_marginal(self):
        prob = []
        states = []
        for chunk in list(self.chunks):
            prob.append(chunk.count)
            states.append(chunk)
        prob = [k / sum(prob) for k in prob]
        return self.sample_from_distribution(states, prob)

    def imagination1d(self, seql = 10):
        ''' Imagination on one dimensional sequence '''
        self.convert_chunks_in_arrays() # convert chunks to np arrays
        img = np.zeros([1,1,1])
        l = 0
        while l< seql:
            chunk,p = self.sample_marginal()
            chunkarray = chunk.arraycontent
            img = np.concatenate((img,chunkarray), axis=0)
            print('sampled chunk array is ', chunkarray, ' p = ', p)
            print('imaginative sequence is ', img)
            l = l + chunkarray.shape[0]
        return img[1:seql, :, :]


    def imagination(self, n, sequential=False, spatial=False, spatial_temporal = False):
        ''' Independently sample from a set of chunks, and put them in the generative sequence
            Obly support transitional probability at the moment
            n+ temporal length of the imagiantion'''

        marginals = self.M
        s_last_index = np.random.choice(np.arange(0,len(list(self.M.keys())),1))
        s_last = list(self.M.keys())[s_last_index]
        # s_last = np.random.choice(list(self.M.keys()))
        s_last = tuple_to_arr(s_last)
        H, W = s_last.shape[1:]
        if sequential:
            L = 20
            produced_sequence = np.zeros([n + L, H, W])
            produced_sequence[0:s_last.shape[0], :, :] = s_last
            t = s_last.shape[0]
            while t <= n:
                #item, p = sample(transition_matrix, marginals, arr_to_tuple(s_last))
                item, p = sample_marginal(marginals)
                s_last = tuple_to_arr(item)
                produced_sequence[t:t + s_last.shape[0], :, :] = s_last
                t = t + s_last.shape[0]
            produced_sequence = produced_sequence[0:n,:,:]
            return produced_sequence

        if spatial:
            produced_sequence = np.zeros([n, H, W])
            produced_sequence[0, :, :] = s_last
            t = 1
            while t <= n:
                # this part is not as simple, because transition matrix is spatial.
                item, p = sample_spatial(marginals)
                s_last = item
                produced_sequence[t:t+1, :, :] = s_last
                t = t + 1
            return produced_sequence

        if spatial_temporal:
            L = 20
            produced_sequence = np.zeros([n+L, H, W])
            produced_sequence[0:s_last.shape[0], :, :] = s_last
            t = 1
            while t <= n:
                # this part is not as simple, because transition matrix is spatial.
                item, p = sample_spatial_temporal(marginals)
                s_last = item
                produced_sequence[t:t+s_last.shape[0], :, :] = s_last
                t = t + s_last.shape[0]
            return produced_sequence

        else:
            return None




