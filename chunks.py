import random
import string
import numpy as np


class Chunk:
    """ Spatial Temporal Chunk
        At the moment, it lacks a unique identifier for which of the chunk is which, making the searching process
        diffidult, ideally, each chunk should have its own unique name, (ideally related to how it looks like) """

    # A code name unique to each chunk
    def __init__(self, chunkcontent, variable=[], count=1, H=None, W=None, pad=1, entailment=[]):
        """chunkcontent: a list of tuples describing the location and the value of observation"""
        # TODO: make sure that there is no redundency variable
        self.content = set(chunkcontent)
        self.key = tuple(sorted(self.content))
        self.count = count  #
        self.T = int(max(np.array(chunkcontent)[:, 0]) + 1)  # those should be specified when joining a chunking graph
        self.H = H
        self.W = W
        self.index = None
        self.vertex_location = None
        self.pad = pad  # boundary size for nonadjacency detection, set the pad to not 1 to enable this feature.
        self.adjacency = {}
        self.birth = None  # chunk creation time
        self.volume = len(self.content)  #
        self.indexloc = self.get_index()
        self.arraycontent = None
        self.boundarycontent = set()
        T, H, W, cRidx = self.get_index_padded()
        self.D = 10
        self.matching_threshold = 0.8
        self.matching_seq = {}
        self.abstraction = []  # what are the variables summarizing this chunk
        self.variable = [] # the variables that this chunk is included as an instance
        self.entailment = entailment  # concrete chunks that the variable is pointing to
        self.cl = {}  # left decendent
        self.cr = {}  # right decendent
        self.acl = {} # left ancestor
        self.acr = {} # right ancestor

        # discount coefficient when computing similarity between two chunks, relative to the temporal discount being 1
        self.h = 1.
        self.w = 1.
        self.v = 1.

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return not(self == other)


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
        assert dt in list(self.adjacency.keys())
        N = 0
        for item in self.adjacency[dt]:
            N = N + self.adjacency[dt][item]
        return N

    def get_index(self):
        ''' Get index location the nonzero chunk locations in chunk content '''
        return set(map(tuple, np.array(list(self.content))[:, 0:3]))

    def get_index_padded(self):
        ''' Get padded index arund the nonzero chunk locations '''
        padded_index = self.indexloc.copy()
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
        for dt in list(self.adjacency.keys()):
            for chunkidx in list(self.adjacency[dt].keys()):
                self.adjacency[dt][chunkidx] = 0
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
            clcrcontent = self.content | cR.content
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
        self.T = int(np.atleast_2d(np.array(list(self.content)))[:,
                     0].max() + 1)  # those should be specified when joining a chunking graph
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

    def check_adjacency(self, cR, dt=0):

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
        # TODO: can simply subtract the coordinate values and check the difference
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

        if (lap_t > 0 & lap_x > 0 & lap_y > 0):
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

    def update_transition(self, chunkkey, dt):
        if chunkkey in self.adjacency.keys():
            if dt in self.adjacency[chunkkey].keys():
                self.adjacency[chunkkey][dt] = self.adjacency[chunkkey][dt] + 1
            else:
                self.adjacency[chunkkey][dt] = 1
        else:
            self.adjacency[chunkkey] = {}
            if dt in self.adjacency[chunkkey].keys():
                self.adjacency[chunkkey][dt] = 1

        for v in self.variable:
            if dt in v.adjacency[chunkkey].keys():
                v.adjacency[chunkkey][dt] = v.adjacency[chunkkey][dt] + 1
            else:
                v.adjacency[chunkkey][dt] = 1
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
    '''TODO: upon initialization, need to inherent previous counts '''
    """A variable can take on several contents"""

    # A code name unique to each chunk
    def __init__(self, entailingchunks, totalcount=1):  # how to define a variable?? a list of alternative
        '''variablecontent: a list of possible content that a variable can take, it can be a list of sets, variables, and chunks '''
        self.content = entailingchunks  # a variable is a pointer to several enlisted chunks
        for chunk in entailingchunks:
            chunk.variable.add(self)
        self.totalcount = totalcount
        self.key = self.get_variable_key()
        self.adjacency = self.get_adjacency(entailingchunks)  # should the adjaceny specific to individual variable instances, or as the entire variable? entire variable.
        self.entailment = {}
        self.volume = len(self.content)  #

        self.arraycontent = None
        self.boundarycontent = set()
        self.D = 1
        self.h = 1.
        self.w = 1.
        self.v = 1.

    def update(self, varinstance):  # when any of its associated chunks are identified
        if varinstance in self.content:
            self.count[varinstance] = self.count[varinstance] + 1
        self.totalcount += 1
        return

    def get_adjacency(self, entailingchunks):
        # I think we might not need it
        adjacency = {}
        dts = set()
        for chunk in entailingchunks:
            for dt in chunk.adjacency:
                for cr in chunk.adjacency[dt]:
                    pass
        return

    def update_transition(self, chunkidx, dt):  # _c_
        # transition can be chunk or variable
        if dt in list(self.adjacency.keys()):
            if chunkidx in list(self.adjacency[dt].keys()):
                self.adjacency[dt][chunkidx] = self.adjacency[dt][chunkidx] + 1
            else:
                self.adjacency[dt][chunkidx] = 1
        else:
            self.adjacency[dt] = {}
            self.adjacency[dt][chunkidx] = 1
        return


    def get_N_transition(self, dt):
        assert dt in list(self.adjacency.keys())
        # todo: make nonozero
        N = 0
        for item in self.adjacency[dt]:
            N = N + self.adjacency[dt][item]
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

    def concatinate(self, cR):
        '''concatinate a variable with a chunk, or a variable with a variable'''
        # TODO: concatination:
        if self.check_adjacency(cR):
            # TODO: modify this part
            clcrcontent = self.content | cR.content
            clcr = Chunk(list(clcrcontent), H=self.H, W=self.W)

        pass
        return

    def get_variable_key(self):
        length = 4
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str