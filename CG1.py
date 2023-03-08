from Learning import *
from chunks import *
import numpy as np
import copy


class CG1:
    """
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
    def __init__(self, y0=0, x_max=0, DT=0.01, theta=0.75):
        """DT: deletion threshold"""
        # vertex_list: list of vertex with the order of addition
        # each item in vertex list has a corresponding vertex location.
        # edge list: list of vertex tuples
        self.vertex_list = []  # list of the chunks
        # the concrete and abstract chunk list together i
        self.y0 = y0  # the initial height of the graph, used for plotting
        self.x_max = x_max  # initial x location of the graph
        self.chunks = {}  # a dictonary with chunk keys and chunk tuples
        self.chunk_probabilities ={}
        self.variables = {}  # variable with their variable object
        self.variablekeys = {} # used for checking variable duplicates
        self.concrete_chunks = {}  # no entailment
        self.ancestors = []  # list of chunks without parents
        self.latest_descendents = [] # chunks without children
        self.theta = theta  # forgetting rate
        self.deletion_threshold = DT
        self.H = 1  # default
        self.W = 1
        self.zero = None
        self.relational_graph = False

    def get_N(self):
        """returns the number of parsed observations"""
        assert len(self.chunks) > 0
        N = 0
        for i in self.chunks:
            N = N + self.chunks[i].count
        return N



    def get_N_transition(self, dt=None):
        """returns the number of parsed observations"""
        assert len(self.chunks) > 0

        if dt == None:
            N_transition = 0
            for chunk in self.chunks:
                for ck in self.chunks[chunk].adjacency:
                    N_transition = N_transition + sum(self.chunks[chunk].adjacency[ck].values())

            return N_transition
        else:
            N_transition = 0
            for chunk in self.chunks:
                for ck in self.chunks[chunk].adjacency:
                    if dt in self.chunks[chunk].adjacency[ck]:
                        N_transition = N_transition + self.chunks[chunk].adjacency[ck][dt]

            return N_transition

    def empty_counts(self):
        """empty count entries and transition entries in each chunk"""
        for ck in self.chunks.values():
            ck.count = 0
            ck.empty_counts()
        for v in self.variables:
            var = self.variables[v]
            var.count = 0
            var.empty_counts()
        return

    def extrapolate_variable(self, chunk):
        transition = chunk.adjacency
        count = 0
        entailment = {}
        for otherchunk in transition:
            for dt in transition[otherchunk]:
                count = count + 1
            entailment[otherchunk] = sum(transition[otherchunk].item())

        Var = Chunk([], variable=True, count=count, H=None, W=None, pad=1, entailment=entailment)
        for ck in entailment:
            ck.abstraction[Var] = sum(chunk.adjacency[ck].item())
        return

    # def abstraction_learning(self, ancestors):
    #     # find all chunks with a common cause or a common effect, extrapolating them into a variable
    #     if ancestors.cl == [] and ancestors.cr == []:
    #         return
    #     elif ancestors.cl != [] and ancestors.cr == []:
    #         abstraction_learning(ancestors.cr)
    #     elif ancestors.cl == [] and ancestors.cr != []:
    #         abstraction_learning(ancestors.cl)
    #     else:
    #         abstraction_learning(ancestors.cl)
    #         abstraction_learning(ancestors.cr)

    def hypothesis_test(self, clidx, cridx, dt):
        if clidx in self.chunks:cl = self.chunks[clidx]
        else: cl = self.variables[clidx]
        if cridx in self.chunks:cr = self.chunks[cridx]
        else: cr = self.variables[cridx]

        assert len(cl.adjacency) > 0
        assert dt in list(cl.adjacency[cridx].keys())
        N = self.get_N()
        N_transition = self.get_N_transition(dt=dt)

        N_min = 5
        # Expected
        ep1p1 = (cl.count / N) * (cr.count / N) * N_transition
        ep1p0 = (cl.count / N) * (N - cr.count) / N * N_transition
        ep0p1 = (N - cl.count) / N * (cr.count / N) * N_transition
        ep0p0 = (N - cl.count) / N * (N - cr.count) / N * N_transition

        # Observed
        op1p1 = cl.adjacency[cridx][dt]
        op1p0 = cl.get_N_transition(dt) - cl.adjacency[cridx][dt]
        op0p1 = 0
        op0p0 = 0
        for ncl in list(self.chunks.values()):  # iterate over p0, which is the cases where cl is not observed
            if ncl != cl:
                if cridx in list(ncl.adjacency.keys()):
                    if dt in list(ncl.adjacency[cridx].keys()):
                        op0p1 = op0p1 + ncl.adjacency[cridx][dt]
                        for ncridx in list(ncl.adjacency.keys()):
                            if self.chunks[ncridx] != cr:
                                op0p0 = op0p0 + ncl.adjacency[ncridx][dt]

        if op0p0 <= N_min and op1p0 <= N_min and op1p1 <= N_min and op0p1 <= N_min:
            return True
        else:
            _, pvalue = stats.chisquare([op1p1, op1p0, op0p1, op0p0], f_exp=[ep1p1, ep1p0, ep0p1, ep0p0], ddof=1)
            # print('p value is ', pvalue)
            if pvalue < 0.05:
                return False  # reject independence hypothesis, there is a correlation
            else:
                return True

    def getmaxchunksize(
            self,
    ):  # TODO: alternatively, update this value upon every chunk creation
        maxchunksize = 0
        if len(self.chunks) > 0:
            for ck_key, ck in self.chunks.items():
                if ck.volume > maxchunksize:
                    maxchunksize = ck.volume

        return maxchunksize

    def observation_to_tuple(self, relevant_observations):
        """relevant_observations: array like object"""
        index_t, index_i, index_j = np.nonzero(relevant_observations)  # observation indexes
        value = [relevant_observations[t, i, j] for t, i, j in zip(index_t, index_i, index_j) if
                 relevant_observations[t, i, j] > 0]
        content = set(zip(index_t, index_i, index_j, value))
        maxT = max(index_t)
        return (content, maxT)

    def update_hw(self, H, W):
        self.H = H
        self.W = W
        return

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
            while len(this_chunk.cl) > 0:  # has children

                if len(this_chunk.cl) == 1:  # only one children
                    if this_chunk.acl == []:  # ancestor node
                        self.ancestors.pop(this_chunk)
                        self.ancestors.__add__(this_chunk.cl)
                    else:
                        ancestor = this_chunk.acl
                        ancestor.cl.add(this_chunk.cl)
                        this_chunk.acr.cr.pop(this_chunk)
                        this_chunk.acr = []
                    for rightkid in this_chunk.cr:
                        this_chunk.cl.__add__(rightkid)
                        rightkid.cl = ancestor  # TODO: can add right chunk ancestor to children as well.
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

    def save_graph(self, name='', path=''):
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
            dictionary[key] = 0
        return dictionary

    def independence_test(self):
        """Evaluate the independence as a stopping criteria for the model"""
        pass
        return False

    def relational_graph_refactorization(self, newc):
        # find biggest common intersections between newc and all other previous chunks in the set
        for chunk in self.vertex_list:
            if chunk.children == []:  # start from the leaf node to find the biggest smaller intersection.
                max_intersect = newc.content.intersection(chunk.content)
                if max_intersect in self.visible_chunk_list:  # create an edge between the biggest smaller intersection and newc
                    idx_max_intersect = self.visible_chunk_list[max_intersect].idx
                    if ~self.check_ancestry(chunk, max_intersect):  # link max intersect to newc
                        self.edge_list.append((idx_max_intersect, self.chunks[newc].idx))
                        self.chunks[idx_max_intersect].children.append(newc)
                    else:  # in chunk's ancestors:
                        print('intersection already exist')
                        self.edge_list.append((idx_max_intersect, self.chunks[newc].idx))
                        chunk.children.append(newc)
                else:  # if not, add link from this chunk to newc and some chunk
                    max_intersect_chunk = Chunk(list(max_intersect), H=chunk.H, W=chunk.W)
                    self.add_chunk(max_intersect_chunk, leftidx=None, rightidx=None)
                    self.edge_list.append((self.chunks[max_intersect_chunk].idx, self.chunks[newc].idx))
                    self.edge_list.append((self.chunks[max_intersect_chunk].idx, self.chunks[chunk].idx))
                    max_intersect_chunk.children.append(newc)
                    max_intersect_chunk.children.append(chunk)
        return

    def check_ancestry(self, chunk, content):
        # check if content belongs to ancestor
        if chunk.parents == []:
            return content != chunk.content
        else:
            return np.any([self.check_ancestry(parent, content) for parent in chunk.parents])

    def update_empty(self, n_empty):
        """chunk: nparray converted to tuple format
        Every time when a new chunk is identified, this function should be called """
        ZERO = self.zero
        self.M[ZERO] = self.M[ZERO] + n_empty
        return

    def check_chunk_in_M(self, chunk):
        """chunk object"""
        # content should be the same set
        for otherchunk in M:
            intersect = otherchunk.intersection(chunk)
            if len(intersect) == chunk.volume:
                return otherchunk
        return

    def check_chunkcontent_in_M(self, chunkcontent):
        if chunkcontent in self.chunks:
            return self.chunks[chunkcontent]
        else:
            return None

    # update graph configuration
    def add_chunk(self, newc, ancestor = False, leftkey=None, rightkey=None):
        if ancestor:
            self.ancestors.append(newc)  # point from content to chunk

        self.vertex_list.append(self.make_hash(newc.ordered_content))
        self.chunks[newc.key] = newc  # add observation

        if len(newc.variables) == 0:  # record concrete chunks and variables separately
            self.concrete_chunks[newc.key] = newc
        else:
            for varc in newc.variables:
                self.variables[varc.key] = varc
                self.variablekeys[varc.entailingchunknames] = varc
                varc.chunks[newc.key] = newc

        newc.H = self.H  # initialize height and weight in chunk
        newc.W = self.W
        # compute the x and y location of the chunk based on pre-existing
        # graph configuration, when this chunk first emerges
        if leftkey is None and rightkey is None:
            x_new_c = self.x_max + 1
            y_new_c = self.y0
            self.x_max = x_new_c
            newc.vertex_location = [x_new_c, y_new_c]
        else:
            if leftkey in self.chunks:leftparent = self.chunks[leftkey]
            else: leftparent = self.variables[leftkey]
            if rightkey in self.chunks:rightparent = self.chunks[rightkey]
            else: rightparent = self.variables[rightkey]
            l_x, l_y = leftparent.vertex_location
            r_x, r_y = rightparent.vertex_location  # [rightkey]

            x_c = (l_x + r_x) * 0.5
            y_c = self.y0
            newc.vertex_location = [x_c, y_c]
            self.y0 = self.y0 + 1

            leftparent.cl = self.check_and_add_to_dict(leftparent.cl, newc)
            rightparent.cr = self.check_and_add_to_dict(rightparent.cr, newc)
            newc.acl = self.check_and_add_to_dict(newc.acl, leftparent)
            newc.acr = self.check_and_add_to_dict(newc.acr, rightparent)

        return

    def add_chunk_to_cg_class(self, chunkcontent):
        """chunk: nparray converted to tuple format
        Every time when a new chunk is identified, this function should be called """
        matchingchunk = self.check_chunkcontent_in_M(chunkcontent)
        if matchingchunk != None:
            self.M[matchingchunk] = self.M[matchingchunk] + 1
        else:
            matchingchunk = Chunk(chunkcontent, H=self.H, W=self.W)  # create an entirely new chunk
            self.M[matchingchunk] = 1
            self.add_chunk(matchingchunk)
        return matchingchunk

    # convert frequency into probabilities

    def forget(self):
        """ discounting past observations if the number of frequencies is beyond deletion threshold"""
        for chunk in self.chunks:
            chunk.count = chunk.count * self.theta
            # if chunk.count < self.deletion_threshold: # for now, cancel deletion threshold, as index to chunk is
            # still vital
            #     self.chunks.pop(chunk)
            for dt in list(chunk.adjacency.keys()):
                for adj in list(chunk.adjacency[dt].keys()):
                    chunk.adjacency[dt][adj] = chunk.adjacency[dt][adj] * self.theta
                    if chunk.adjacency[dt][adj] < self.deletion_threshold:
                        chunk.adjacency[dt].pop(adj)

        return


    def checkcontentoverlap(self, chunk):
        '''check if the content is already contained in one of the chunks'''
        try:
            if len(chunk.variables) ==0:
                if chunk.key in self.chunks:
                    return self.chunks[chunk.key]
                else:
                    return None
            else: # this chunk does contain variable component:
                hashed_content = self.make_hash(chunk.ordered_content)
                if hashed_content in self.chunks:
                    return self.chunks[hashed_content]
        except(AttributeError):
            print('')

            


    def chunking_reorganization(self, prevkey, currentkey, cat, dt):
        '''dt: end_prev(inclusive) - start_post(exclusive)'''
        def findancestors(c, L):
            '''Find all ancestors of c'''
            if len(c.acl) == 0:
                return
            else:
                L = L + list(c.acl.keys())
                for i in list(c.acl.keys()):
                    findancestors(i, L)
        if prevkey in self.chunks: prev = self.chunks[prevkey]
        else: prev = self.variables[prevkey]
        if currentkey in self.chunks: current = self.chunks[currentkey]
        else: current = self.variables[currentkey]
        chunk = self.checkcontentoverlap(cat)
        if chunk is None:  # add concatinated chunk to the network
            # TODO： add chunk to vertex
            self.add_chunk(cat, leftkey=prevkey, rightkey=currentkey)
            # iterate through all chunk transitions that could lead to the same concatination chunk
            cat.count = prev.adjacency[currentkey][dt]  # need to add estimates of how frequent the joint frequency occurred
            prev.count = prev.count - cat.count  # reduce one sample observation from the previous chunk
            current.count = current.count - cat.count
            prev.adjacency[current.key][dt] = 0

            cat.adjacency = copy.deepcopy(current.adjacency)
            # check other pathways that arrive at the same chunk based on cat's ancestor
            candidate_cls = []
            findancestors(cat, candidate_cls)  # look for all ancestoral graph path that arrive at cat
            for ck in candidate_cls:
                for _cr in ck.adjacency:
                    for _dt in ck[_cr]:
                        if _cr != currentkey and ck.key != prevkey and _dt != dt:
                            _cat = combinechunks(ck.key, _cr, _dt)
                            if _cat != None:
                                if _cat == cat:
                                    # TODO: may need to merge nested dictionary
                                    _cat_count = self.chunks[ck].adjacency[_cr][_dt]
                                    cat.count = cat.count + _cat_count
                                    ck.count = ck.count - _cat_count
                                    _cr.count = _cr.count - _cat_count
                                    ck.adjacency[_cr][_dt] = 0

                                    leftparent = self.chunks[ck.key]
                                    rightparent = self.chunks[_cr]
                                    leftparent.cl = self.check_and_add_to_dict(leftparent.cl, cat)
                                    rightparent.cr = self.check_and_add_to_dict(rightparent.cr, cat)
                                    cat.acl = self.check_and_add_to_dict(cat.acl, leftparent)
                                    cat.acr = self.check_and_add_to_dict(cat.acl, rightparent)
        else:
            chunk.count = chunk.count + prev.adjacency[currentkey][dt]  # need to add estimates of how frequent the joint frequency occurred
            prev.count = prev.count - cat.count  # reduce one sample observation from the previous chunk
            current.count = current.count - cat.count
            prev.adjacency[current.key][dt] = 0

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
        v = 3  # threshold of volume of intersection
        app_t = 3  # applicability threshold
        '''cat: new chunk which just entered into the system
        find the intersection of cat with the pre-existing chunks '''
        # (content of intersection, their associated chunks) ranked by the applicability threshold
        # alternatively, the most applicable intersection:
        max_intersect = None
        max_intersect_count = 0 #
        max_intersect_chunks = []  # chunks that needs to be merged
        for ck in self.chunks:
            if ck.variables==[] and cat.variables == []:
                intersect = ck.content.intersection(cat.content)
                intersect_chunks = []
                c = 0  # how often this intersection is applicable across chunks
                if len(intersect) != len(cat.content) and len(intersect) > v:  # not the same chunk
                    # look for overlap between this intersection and other chunks:
                    for ck_ in self.chunks:  # how applicable is this intersection, to other previously learned chunks
                        if ck_.content.intersection(intersect) == intersect:
                            c = c + ck_.count
                            intersect_chunks.append(ck_)
                if c > max_intersect_count and c >= app_t:
                    # atm. select intersect with the max intersect count
                    # TODO: can be ranked based on merging gain
                    max_intersect_count = c
                    max_intersect_chunks = intersect_chunks
                    max_intersect = intersect
            elif len(ck.variables)>0 and len(cat.variables)>0: # both of the chunks contain variables
                intersect = LongComSubS(list(ck.ordered_content), list(cat.ordered_content))
                c = 0  # how often this intersection is applicable across chunks
                intersect_chunks = [] # a list of string denoting chunk content and variable name
                # TODO: need to look at ck.ordered_content - intersect, to become variables, I will leave this part for now.
                if len(intersect) != len(cat.content) and len(intersect) > v:  # not the same chunk
                    for ck_ in self.chunks:
                        if LongComSubS(list(ck_.ordered_content), intersect):
                            c = c + ck_.count
                            intersect_chunks.append(ck_)
                if c > max_intersect_count and c >= app_t:
                    # atm. select intersect with the max intersect count
                    # TODO: can be ranked based on merging gain
                    max_intersect_count = c
                    max_intersect_chunks = intersect_chunks
                    max_intersect = intersect

        if max_intersect != None:  # reorganize chunk list to integrate with variables
            self.merge_chunks(max_intersect, max_intersect_chunks, max_intersect_count)
        return


    def LongComSubS(self,st1, st2):
        '''Longest Common Substring, used to check chunk overlaps'''
        # TODO: feel like this can be replaced with something smarter
        maxsize = 0
        maxsubstr = []
        for a in range(len(st1)):
            for b in range(len(st2)):
                k = 0
                substr = []
                while ((a + k) < len(st1) and (b + k) < len(st2) and st1[a + k] == st2[b + k]):
                    k = k + 1
                    substr.append(st1[a + k])
                if k >=maxsize:
                    maxsize = k
                    maxsubstr = substr
        return maxsubstr


    def merge_chunks(self, max_intersect, max_intersect_chunks, max_intersect_count):
        # create a new chunk with intergrated variables.
        # for ck in max_intersect_chunks: # for now, do not touch the content within each chunk, but create a separate abstraction node
        #     ck.content = ck.content - max_intersect
        self.set_variable_adjacency(var, max_intersect_chunks)
        if isinstance(max_intersect, set):
            chk = None  # find if intersection chunk exists in the past
            if tuple(sorted(self.content)) in self.chunks.keys():
                chk = self.chunks[tuple(sorted(self.content))]
            if chk == None:  # add new chunk here
                chk = Chunk(max_intersect, count=max_intersect_count)
            else:
                assert (max_intersect_count > chk.count)
                chk.count = max_intersect_count

            for ck in max_intersect_chunks: # update graph relation
                chk.entailment[ck.key] = ck
                ck.abstraction[chk.key] = chk
        else: # max_intersect is a list
            chk = None
            dictionary = {}
            for i in range(0, len(max_intersect)):
                dictionary[i] = max_intersect[i]
            hashed_dictionary = self.make_hash(dictionary)
            if hashed_dictionary in self.chunks.keys():
                chk = self.chunks[hashed_dictionary]

            if chk == None:  # add new chunk here
                chk = Chunk([], count=max_intersect_count)
                chk.ordered_content = dictionary
            else:
                assert (max_intersect_count > chk.count)
                chk.count = max_intersect_count

            for ck in max_intersect_chunks: # update graph relation
                chk.entailment[ck.key] = ck
                ck.abstraction[chk.key] = chk
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

    def imagination1d(self, seql=10):
        ''' Imagination on one dimensional sequence '''
        self.convert_chunks_in_arrays()  # convert chunks to np arrays
        img = np.zeros([1, 1, 1])
        l = 0
        while l < seql:
            chunk, p = self.sample_marginal()
            chunkarray = chunk.arraycontent
            img = np.concatenate((img, chunkarray), axis=0)
            print('sampled chunk array is ', chunkarray, ' p = ', p)
            print('imaginative sequence is ', img)
            l = l + chunkarray.shape[0]
        return img[1:seql, :, :]

    def abstraction_learning(self):
        """
        Create variables from adjacency matrix.
        variable construction: chunks that share common ancestors and common descendents.
        pre---variable---post, for each dt time: variables with common cause and common effect
        """
        # TODO: another version with more flexible dt
        varchunks_to_add = []
        T = 1
        for chunk in self.chunks.values(): #latestdescendents
            v_horizontal_ = set(chunk.adjacency.keys())
            # TODO: need to consider different times in the future
            for postchunk in self.chunks.values():# latestdescedents
                v_vertical_ = set(postchunk.preadjacency.keys())
                candidate_variable_entailment = v_horizontal_.intersection(v_vertical_)
                if len(candidate_variable_entailment) > T: #register a variable
                    candidate_variables = {self.chunks[candidate] for candidate in candidate_variable_entailment}
                    v = Variable(candidate_variables, self)
                    v = self.add_variable(v, candidate_variable_entailment)
                    # create variable chunk: chunk + var + postchunk
                    # need to roll it out when chunk itself contains variables.
                    ordered_content = chunk.ordered_content.copy()
                    ordered_content.append(v.key)
                    ordered_content = ordered_content + postchunk.ordered_content
                    var_chunk = Chunk(([]), variables = set([v]), ordered_content=ordered_content)
                    varchunks_to_add.append(var_chunk)

        for var_chunk in varchunks_to_add:
            self.add_chunk(var_chunk)
        return

    def check_variable_duplicates(self,newv):
        try:
            if newv.entailingchunknames in set(list(self.variablekeys.keys())):
                return True # duplicate
            else:
                return False
        except(TypeError):
            print('')


    def add_variable(self, v, candidate_variable_entailment):
        if self.check_variable_duplicates(v) == False:  # check duplicates
            self.variables[v.key] = v  # update chunking graph
            self.variablekeys[v.entailingchunknames] = v
            for ck in candidate_variable_entailment:
                self.chunks[ck].variables[v.key] = v
            return v
        else:
            return self.variablekeys[v.entailingchunknames]


    def make_hash(self,o):
      """
      Makes a hash from a dictionary, list, tuple or set to any level, that contains
      only other hashable types (including any lists, tuples, sets, and
      dictionaries).
      """
      if isinstance(o, (set, tuple, list)):
        return tuple([self.make_hash(e) for e in o])
      elif not isinstance(o, dict):
        return hash(o)
      new_o = copy.deepcopy(o)
      for k, v in new_o.items():
        new_o[k] = self.make_hash(v)

      return hash(tuple(frozenset(sorted(new_o.items()))))