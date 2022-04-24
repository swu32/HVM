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
        self.vertex_location = []
        self.visible_chunk_list = [] # list of visible chunks # the representation of absract chunk entries is replaced by 0
        # the concrete and abstract chunk list together i
        self.edge_list = []
        self.y0 = y0 # the initial height of the graph, used for plotting
        self.x_max = x_max # initial x location of the graph
        self.chunks = []# a dictonary with chunk keys and chunk tuples,
        self.concrete_chunks = []# list of chunks with no variables
        self.abstract_chunks = []# list of chunks with variables
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
        for ck in self.chunks:
            N = N + ck.count
        return N


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


    def prediction(self,seq):
        # TODO: how does the prediction work?
        """input: seq in np array
           output: prediction of the next several sequential items based on the previous observations,
           and the confidence judged by the model"""
        prediction = {}
        return prediction


    # update graph configuration
    def add_chunk(self, newc, leftidx= None, rightidx = None):
        # TODO: add time when the chunk is being created
        self.vertex_list.append(newc)
        self.chunks.append(newc) # add observation
        self.visible_chunk_list.append(newc.content)
        newc.index = self.chunks.index(newc)
        newc.H = self.H # initialize height and weight in chunk
        newc.W = self.W
        # compute the x and y location of the chunk based on pre-existing
        # graph configuration, when this chunk first emerges
        if leftidx is None and rightidx is None:
            x_new_c = self.x_max + 1
            y_new_c = self.y0
            self.x_max = x_new_c
            self.vertex_location.append([x_new_c, y_new_c])
        else:
            l_x, l_y = self.vertex_location[leftidx]
            r_x, r_y = self.vertex_location[rightidx]
            x_c = (l_x + r_x)*0.5
            y_c = self.y0
            self.y0 = self.y0 + 1
            idx_c = len(self.vertex_list) - 1
            self.edge_list.append((leftidx, idx_c))
            self.edge_list.append((rightidx, idx_c))
            self.vertex_location.append([x_c, y_c])

        #
        # if self.relational_graph:
        #     self.relational_graph_refactorization(newc)

        #TODO: integrate with chunk element classes, if possible
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
        if len(self.M) == 0:
            return None
        else:

            for chunk in list(self.M.keys()):
                if len(chunk.content.intersect(chunkcontent)) == len(chunkcontent):
                    return chunk
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
        for chunk in self.chunks:
            if chunk.contentagreement(content):#
                return chunk
        return None

    def checkcoincidingconcat(self):
        # iterate through all combinations of adjacency matrix
        return

    def chunking_reorganization(self, previdx, currentidx, cat, dt):
        ''' Reorganize marginal and transitional probability matrix when a new chunk is created by concatinating prev and current '''
        prev = self.chunks[previdx]
        current = self.chunks[currentidx]
        """Model hasn't seen this chunk before:"""
        chunk = self.checkcontentoverlap(cat.content)
        if chunk is None:
            # add concatinated chunk to the network
            # TODO： add chunk to vertex
            self.add_chunk(cat, leftidx=previdx, rightidx=currentidx)
            # iterate through all chunk transitions that could lead to the same concatination chunk
            cat.count = prev.adjacency[dt][currentidx] # need to add estimates of how frequent the joint frequency occurred
            cat.adjacency = copy.deepcopy(current.adjacency)
            # iterate through chunk organization to see if there are other pathways that arrive at the same chunk
            for _prevck in self.chunks:
                _previdx = _prevck.index
                for _dt in list(_prevck.adjacency.keys()):
                    for _postidx in list(_prevck.adjacency[_dt].keys()):
                        _postck = self.chunks[_postidx]
                        if _previdx != previdx and _postidx != currentidx:
                            _cat = combinechunks(_previdx, _postidx, _dt, self)
                            if _cat != None:
                                if _cat.contentagreement(cat.content): # the same chunk
                                    cat.count = cat.count + self.chunks[_previdx].adjacency[_dt][_postidx]
                                    self.chunks[_previdx].adjacency[_dt][_postidx] = 0
                                    # TODO: transition should also be updated accordingly
        else:
            chunk.count = chunk.count + prev.adjacency[dt][currentidx]  # need to add estimates of how frequent the joint frequency occurred

        # prev.adjacency[dt][currentidx] = prev.adjacency[dt][currentidx] - 1
        prev.adjacency[dt][currentidx] = 0
        prev.count = prev.count - 1
        current.count = current.count - 1
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

    def imagination(self, n, sequential = False, spatial = False,spatial_temporal = False):
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
            produced_sequence = np.zeros([n+ L, H, W])
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