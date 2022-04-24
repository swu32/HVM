import numpy as np
class Variable:
    """A variable can take on several contents"""

    # A code name unique to each chunk
    def __init__(self, chunkcontent, H = None,W = None):
        self.content = set(chunkcontent)
        self.T = int(max(np.array(chunkcontent)[:, 0])+1) # those should be specified when joining a chunking graph
        self.H = H
        self.W = W
        self.count = 1 #
        self.pad = 10 # boundary size for nonadjacency detection, set the pad to not 1 to enable this feature.
        self.adjacency = {}
        self.volume = len(self.content) #
        self.indexloc = None
        self.get_index()
        self.arraycontent = None
        self.boundarycontent = set()
        T,H,W,cRidx = self.get_index_padded()
        self.D = 1
        self.matching_threshold = 0.8
        self.matching_seq = {}
        self.parent = []
        self.children = []

        # discount coefficient when computing similarity between two chunks, relative to the temporal discount being 1
        self.h = 1.
        self.w = 1.
        self.v = 1.
