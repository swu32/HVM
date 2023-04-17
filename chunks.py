import random
import string
import numpy as np


class Chunk:
    """ Spatial Temporal Chunk
        At the moment, it lacks a unique identifier for which of the chunk is which, making the searching process
        difficult, ideally, each chunk should have its own unique name, (ideally related to how it looks like) """

    # A code name unique to each chunk
    def __init__(self, chunkcontent, variables={}, ordered_content = None, count=0, H=1, W=1, pad=1, entailment = []):
        """chunkcontent: a list of tuples describing the location and the value of observation"""
        # TODO: make sure that there is no redundency variable
        ########################### Content and Property ########################
        #self.current_chunk_content() # dynamic value, to become the real content for variable representations

        if ordered_content!=None:
            self.ordered_content = ordered_content
            self.key = ''
            for item in ordered_content:
                eachcontent = item
                if type(eachcontent) == str:
                    self.key = self.key + eachcontent
                else:
                    self.key = self.key + str(tuple(sorted(eachcontent)))
        else:
            self.ordered_content = [set(chunkcontent)] #specify the order of chunks and variables
            self.key = tuple(sorted(chunkcontent))

        if len(variables)==0: self.variables = {}
        else:
            self.variables = variables
            for key, var in self.variables.items():
                var.chunks[self.key] = self

        self.content = self.get_content(self.ordered_content)
        self.volume = sum([len(chunkcontent) for chunkcontent in self.ordered_content])
        #self.T = sum([int(max(np.array(chunkcontent)[:, 0]) + 1) for chunkcontent in self.ordered_content.values()])  # those should be specified when joining a chunking graph
        self.T = 0 if chunkcontent == list([]) else int(max(np.atleast_2d(np.array(list(self.content)))[:, 0]) + 1) # TODO: fix summation problem
        self.H = H
        self.W = W
        self.vertex_location = None
        self.pad = pad  # boundary size for nonadjacency detection, set the pad to not 1 to enable this feature.
        self.count = count  #
        self.birth = None  # chunk creation time

        ########################### Relational Connection ########################
        self.adjacency = {} # chunk --> something
        self.preadjacency = {} # something --> chunk
        self.indexloc = self.get_index() # TODO: index location
        self.arraycontent = None
        self.boundarycontent = set()

        self.abstraction = []  # what are the variables summarizing this chunk
        self.entailment = entailment  # concrete chunks that the variable is pointing to
        self.cl = {}  # left decendent
        self.cr = {}  # right decendent
        self.acl = {} # left ancestor
        self.acr = {} # right ancestor

        ###################### discount coefficient for similarity computation ########################
        # T, H, W, cRidx = self.get_index_padded() # TODO: index location
        self.D = 10
        self.matching_threshold = 0.8
        self.matching_seq = {}
        self.h = 1.
        self.w = 1.
        self.v = 1.
        self.parse = 0

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return not(self == other)


    def get_random_name(self):
        length = 4
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def generate_content(self, seqc):
        # look for content in itself that matches the sequence
        # for i in self.ordered_content:
        pass

    def get_content(self,ordered_content):
        # ordered_content: an ordered list with the
        if len(self.ordered_content)==1:
            return self.ordered_content[0]
        else:
            pass
            # return the variable instantiated content
            return

    def get_full_content(self):
        '''returns a list of all possible content that this chunk can take'''
        self.possible_path = []
        self.get_content_recursive(self, [])
        return self.possible_path

    def get_content_recursive(self, node, path):
        path = path + list(node.content)
        if len(list(node.variable)) == 0:
            self.possible_path.append(path)
            return
        else:
            for Var in node.variable:
                self.get_content_recursive(Var, path)


    def update_variable_count(self):
        for ck in self.abstraction:
            ck.update()
        return

    def update(self):
        self.count = self.count + 1
        if len(self.abstraction) > 0:
            self.update_variable_count()  # update the count of the subordinate chunks
        return

    def to_array(self):
        '''convert the content into array'''
        # TODO: correct self.T throughout the program
        arrep = np.zeros((int(max(np.atleast_2d(np.array(list(self.content)))[:, 0]) + 1), self.H, self.W))
        for t, i, j, v in self.content:
            arrep[t, i, j] = v
        self.arraycontent = arrep
        return

    def get_N_transition(self, dt):
        N = 0
        for chunk in self.adjacency:
            if dt in list(self.adjacency[chunk].keys()):
                N = N + self.adjacency[chunk][dt]
        return N

    def get_index(self):
        ''' Get index location of the concrete chunks in chunk content, variable index is not yet integrated '''
        if len(self.ordered_content)==0:
            return set()
        elif len(self.variables) >0:
            return set() # give up when there are variables in the sequence
        else:
            # TODO: integrate with ordered chunkcontent
            index_set = set()

            index0 = set(map(tuple, np.atleast_2d(list(self.ordered_content[0]))[:, 0:3]))
            try:
                t_shift = int(np.atleast_2d(list(self.ordered_content[0]))[:, 0].max() + 1)  # time shift is the biggest value in the 0th dimension
            except(TypeError):
                print('')
            index_set.update(index0)
            for item in self.ordered_content[1:]:
                if type(item)!= str:
                    index = set(map(tuple, np.atleast_2d(list(item))[:, 0:3]))
                    shifted_index = self.timeshift(index, t_shift)
                    index_set.update(shifted_index)
                    t_shift = int(np.atleast_2d(list(item))[:, 0].max() + 1)# time shift is the biggest value in the 0th dimension

            return index_set

    def timeshift(self, content, t):
        shiftedcontent = []
        for tup in list(content):
            lp = list(tup)
            lp[0] = lp[0] + t
            shiftedcontent.append(tuple(lp))
        return set(shiftedcontent)

    def get_index_padded(self):
        # TODO: integrate with ordered chunkcontent

        ''' Get padded index arund the nonzero chunk locations '''
        try:
            padded_index = self.indexloc.copy()
        except(AttributeError):
            print('nonetype')
        chunkcontent = self.content
        self.boundarycontent = set()
        T, H, W = self.T, self.H, self.W
        for t, i, j, v in chunkcontent:
            point_pad = {(t + 1, i, j), (t - 1, i, j), (t, min(i + 1, H), j), (t, max(i - 1, 0), j),
                         (t, i, min(j + 1, W)), (t, i, max(j - 1, 0))}

            if point_pad.issubset(self.indexloc) == False:  # the current content is a boundary element
                self.boundarycontent.add((t, i, j, v))
            padded_index = padded_index.union(point_pad)

        if self.pad > 1:  # pad extra layers around the chunk observations
            # there is max height, and max width, but there is no max time.
            for p in range(2, self.pad + 1):
                for t, i, j, v in chunkcontent:
                    padded_boundary_set = {(t + p, i, j), (t - p, i, j), (t, min(i + p, H), j),
                                           (t, max(i - p, 0), j), (t, i, min(j + p, W)), (t, i, max(j - p, 0))}
                    padded_index = padded_index.union(padded_boundary_set)
        return T, H, W, padded_index

    def conflict(self, c_):
        # TODO: check if the contents are conflicting.
        return False

    def empty_counts(self):
        self.count = 0
        self.birth = None  # chunk creation time
        # empty transitional counts
        for chunkidx in list(self.adjacency.keys()):
            for dt in list(self.adjacency[chunkidx].keys()):
                self.adjacency[chunkidx][dt] = 0
        for chunkidx in list(self.preadjacency.keys()):
            for dt in list(self.preadjacency[chunkidx].keys()):
                self.preadjacency[chunkidx][dt] = 0
        return

    def concatinate(self, cR, check=True):
        if check:
            if self.check_adjacency(cR):
                clcrcontent = self.content | cR.content
                clcr = Chunk(list(clcrcontent), H=self.H, W=self.W)
                return clcr
            else:
                return None

        else:
            clcrcontent = self.ordered_content[0] | cR.ordered_content[0]
            clcr = Chunk(list(clcrcontent), H=self.H, W=self.W)
            return clcr

            # concatinate cL with cR:

    def average_content(self):
        # average the stored content with the sequence
        # calculate average deviation
        averaged_content = set()
        assert (len(self.matching_seq) > 0)
        for m in list(self.matching_seq.keys()):  # iterate through content points
            thispt = list(m)
            n_pt = len(self.matching_seq[m])
            otherpt0 = 0
            otherpt1 = 0
            otherpt2 = 0
            otherpt3 = 0
            for pt in self.matching_seq[m]:
                otherpt0 += pt[0] - thispt[0]
                otherpt1 += pt[1] - thispt[1]
                otherpt2 += pt[2] - thispt[2]
                otherpt3 += pt[3] - thispt[3]
            count = max(1, self.count)
            thispt[0] = int(thispt[0] + 1 / count * otherpt0 / n_pt)
            thispt[1] = int(thispt[1] + 1 / count * otherpt1 / n_pt)
            thispt[2] = int(thispt[2] + 1 / count * otherpt2 / n_pt)
            thispt[3] = int(thispt[3] + 1 / count * otherpt3 / n_pt)  # TODO: content may not need to be averaged

            if np.any(thispt) < 0:
                print("")
            averaged_content.add(tuple(thispt))

        self.content = averaged_content
        self.T = int(np.atleast_2d(np.array(list(self.content)))[:,0].max() + 1)  # those should be specified when joining a chunking graph
        self.get_index()
        self.get_index_padded()  # update boundary content
        return

    def variable_check_match(self, seq):  # a sequence matches any of its instantiated variables
        '''returns true if the sequence matches any of the variable instantiaions
        TODO: test this function with variables '''
        if len(self.variable) == 0:
            return self.check_match(seq)
        else:
            match = []
            for ck in self.variable:
                match.append(ck.variable_check_match(seq))
            return any(match)
        
        

    def check_match(self, seq):
        ''' Check explicit content match'''
        self.matching_seq = {}  # free up memory
        # key: chunk content, value: matching points
        # TODO: one can do better with ICP or Point Set Registration Algorithm.
        D = self.D

        def dist(m, pt):
            return (pt[0] - m[0]) ** 2 + (pt[1] - m[1]) ** 2 + (pt[2] - m[2]) ** 2 + (pt[3] - m[3]) ** 2

        def point_approx_seq(m, seq):  # sequence is ordered in time
            for pt in seq:
                if dist(m, pt) <= D:
                    if m in self.matching_seq.keys():
                        self.matching_seq[m].append(pt)
                    else:
                        self.matching_seq[m] = [pt]
                    return True
            return False

        n_match = 0
        for obs in list(self.content):  # find the number of observations that are close to the point
            if point_approx_seq(obs, seq):  # there exists something that is close to this observation in this sequence:
                n_match = n_match + 1

        if n_match / len(self.content) > self.matching_threshold:
            return True  # 80% correct
        else:
            return False
        



    def check_adjacency(self, cR):
        # dt: start_post - start_prev
        """Check if two chunks overlap/adjacent in their content and location"""
        cLidx = self.indexloc
        _,_,_, cRidx = cR.get_index_padded()
        intersect_location = cLidx.intersection(cRidx)
        if (
            len(intersect_location) > 0
        ):  # as far as the padded chunk and another is intersecting,
            return True
        else:
            return False

    def check_adjacency_approximate(self, cR, dt=0):
        # problematic implementation based on min and max of the boundaries.
        def overlaps(a, b):
            """
            Return the amount of overlap,
            between a and b. Bounds are exclusive.
            If >0, the number of bp of overlap
            If 0,  they are book-ended.
            If <0, the distance in bp between them
            """
            return min(a[1], b[1]) - max(a[0], b[0]) - 1

        """Check if two chunks overlap/adjacent in their content and location"""
        cLidx = self.indexloc
        cRidx = cR.indexloc
        intersect_location = cLidx.intersection(cRidx)

        Mcl, Mcr = np.array(list(cLidx)), np.array(list(cRidx))

        tl1, xl1, yl1 = Mcl.min(axis=0)
        tl2, xl2, yl2 = Mcl.max(axis=0)

        tr1, xr1, yr1 = Mcr.min(axis=0)
        tr2, xr2, yr2 = Mcr.max(axis=0)

        lap_t = overlaps((tl1 - self.pad, tl2 + self.pad), (dt + tr1 - self.pad, dt + tr2 + self.pad))
        lap_x = overlaps((xl1 - self.pad, xl2 + self.pad), (xr1 - self.pad, xr2 + self.pad))
        lap_y = overlaps((yl1 - self.pad, yl2 + self.pad), (yr1 - self.pad, yr2 + self.pad))

        if (lap_t > 0 and lap_x > 0 and lap_y > 0):
            return True
        else:
            return False

    def checksimilarity(self, chunk2):
        '''returns the minimal moving distance from point cloud chunk1 to point cloud chunk2'''
        pointcloud1, pointcloud2 = self.content.copy(), chunk2.content.copy()
        lc1, lc2 = len(pointcloud1), len(pointcloud2)
        # smallercloud = [pointcloud1,pointcloud2][np.argmin([lc1,lc2])]
        # match by minimal distance
        match = []
        minD = 0
        for x1 in pointcloud1:
            mindist = 1000
            minmatch = None
            # search for the matching point with the minimal distance
            if len(match) == min(lc1, lc2):
                break
            for x2 in pointcloud2:
                D = self.pointdistance(x1, x2)
                if D < mindist:
                    minmatch = (x1, x2)
                    mindist = D
            match.append(minmatch)
            minD = minD + mindist
            pointcloud2.pop(minmatch[1])
        return minD

    def pointdistance(self, x1, x2):
        ''' calculate the the distance between two points '''
        D = (x1[0] - x2[0]) * (x1[0] - x2[0]) + self.h * (x1[1] - x2[1]) * (x1[1] - x2[1]) + self.w * (
                    x1[2] - x2[2]) * (x1[2] - x2[2]) + self.v * (x1[0] - x2[0]) * (x1[0] - x2[0])
        return D

    def update_transition(self, chunk, dt):
        '''Update adjacency matrix connecting self to adjacent chunks with time distance dt
        Also update the adjacenc matrix of variables '''
        if chunk.key in self.adjacency.keys():
            if dt in self.adjacency[chunk.key].keys():
                self.adjacency[chunk.key][dt] = self.adjacency[chunk.key][dt] + 1
            else:
                self.adjacency[chunk.key][dt] = 1
        else:
            self.adjacency[chunk.key] = {}
            self.adjacency[chunk.key][dt] = 1

        if self.key in list(chunk.preadjacency.keys()): # preadjacency: something --> chunkidx
            if dt in list(chunk.preadjacency[self.key].keys()):
                chunk.preadjacency[self.key][dt] = chunk.preadjacency[self.key][dt] + 1
            else:
                chunk.preadjacency[self.key][dt] = 1
        else:
            chunk.preadjacency[self.key] = {}
            chunk.preadjacency[self.key][dt] = 1

        for v in self.variables.values():
            if dt in v.adjacency[chunk.key].keys():
                v.adjacency[chunk.key][dt] = v.adjacency[chunk.key][dt] + 1
            else:
                v.adjacency[chunk.key][dt] = 1
        return

    def contentagreement(self, content):
        if len(self.content) != len(content):
            return False
        else:  # sizes are the same
            return len(self.content.intersection(content)) == len(content)


# TODO: upon parsing, isinstance(51,Chunk) can be used to check whether something is a chunk or a variable

import random
import string


class Variable():
    """A variable can take on several contents"""

    # A code name unique to each variable
    def __init__(self, entailingchunks, count=1):  # how to define a variable?? a list of alternative
        ##################### Property Parameter ######################
        self.count = self.get_count(entailingchunks)
        self.key = self.get_variable_key()
        self.current_content = None # dynamic value, any of the entailing chunks that this variable is taking its value in
        self.entailingchunknames = self.getentailingchunknames(entailingchunks)
        ##################### Relational Parameter ######################
        self.adjacency = self.get_adjacency(entailingchunks)#should the adjaceny specific to individual variable instances, or as the entire variable? entire variable.
        self.entailingchunks = entailingchunks
        self.chunks = {} # chunks that this variable is a part of
        self.chunk_probabilities = {}
        self.ordered_content = [self.key] #specify the order of chunks and variables
        self.vertex_location = self.get_vertex_location(entailingchunks)


        # There is only variable relationship, but no relationship between variable and left/right parent/child,
        self.boundarycontent = set()
        self.D = 1
        self.h = 1.
        self.w = 1.
        self.v = 1.

        self.cl = {}  # left decendent
        self.cr = {}  # right decendent
        self.acl = {} # left ancestor
        self.acr = {} # right ancestor


    def sample_current_content(self):
        '''sample one of the entailing chunks as the current content of the variable'''
        self.current_content = np.random.choice(list(self.chunk_probabilities.keys()), 1, p = list(self.chunk_probabilities.values()))
        return


    def get_count(self, entailingchunks):
        count = 0
        for ck in entailingchunks:
            count = count + ck.count
        return count

    def update(self, varinstance):  # when any of its associated chunks are identified
        if varinstance in self.content:
            self.count[varinstance] = self.count[varinstance] + 1
        self.totalcount += 1
        return

    def get_vertex_location(self, entailingchunks):
        xs = 0
        ys = 0
        for ck in entailingchunks:
            x,y = ck.vertex_location
            xs = xs + x
            ys = ys + y
        return xs/len(entailingchunks), ys/len(entailingchunks)

    def get_adjacency(self, entailingchunks):
        # I think we might not need it
        adjacency = {}
        dts = set()
        # entailingchunks = set(cg.chunks[item] for item in entailingchunks)
        entailingchunks = set(entailingchunks)
        for chunk in entailingchunks:
            for cr in chunk.adjacency:
                if cr in adjacency.keys():
                    for dt in chunk.adjacency[cr]:
                        if dt in list(adjacency[cr].keys()):
                            adjacency[cr][dt] = adjacency[cr][dt] + chunk.adjacency[cr][dt]
                        else:
                            adjacency[cr][dt] = chunk.adjacency[cr][dt]
                else:
                    adjacency[cr] = {}
                    for dt in chunk.adjacency[cr]:
                        adjacency[cr][dt] = chunk.adjacency[cr][dt]
        return adjacency

    # def update_transition(self, chunkidx, dt):  # _c_
    #     # transition can be chunk or variable
    #     if chunkidx in list(self.adjacency.keys()):
    #         if dt in list(self.adjacency[chunkidx].keys()):
    #             self.adjacency[chunkidx][dt] = self.adjacency[chunkidx][dt] + 1
    #         else:
    #             self.adjacency[chunkidx][dt] = {}
    #             self.adjacency[chunkidx][dt] = 1
    #     else:
    #         self.adjacency[chunkidx] = {}
    #         self.adjacency[chunkidx][dt] = 1

    def getentailingchunknames(self, entailingchunks):
        """get the content of the entailing chunks"""
        chunknames = set()
        for ck in entailingchunks:
            chunknames.add(ck.key)
        return tuple(list(chunknames))


    def get_N_transition(self, dt):
        # todo: make nonozero
        N = 0
        for chunk in self.adjacency:
            if dt in self.adjacency[chunk]:
                N = N + self.adjacency[chunk][dt]
        return N

    def check_adjacency(self, cR):
        """ Check the adjacency between variable and cR as an observation """
        return any([_ck.check_adjacency(_ck, cR) for _ck in self.content])

    def check_match(self, seq):
        """check whether this variable is in sequence"""
        return any([_ck.check_match(_ck, seq) for _ck in self.content])

    def contentagreement(self, content):
        # if the content agree with any of the chunks within the varible
        pass

    def empty_counts(self):
        self.count = 0
        self.birth = None  # chunk creation time
        # empty transitional counts
        for chunkidx in list(self.adjacency.keys()):
            for dt in list(self.adjacency[chunkidx].keys()):
                self.adjacency[chunkidx][dt] = 0
        for chunkidx in list(self.preadjacency.keys()):
            for dt in list(self.preadjacency[chunkidx].keys()):
                self.preadjacency[chunkidx][dt] = 0
        return


    def get_variable_key(self):
        length = 4
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def check_variable_match(self, seqc):
        '''Check any of variables included in the chunk is consistent with observations of the sequence copy'''
        # for obj in self.content: # obj can be chunk, or variable
        pass

