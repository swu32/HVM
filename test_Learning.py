from unittest import TestCase


class Test(TestCase):
    def test_hcm_learning(self):

        parsingtime = time.perf_counter() - currenttime
        print('elapsed time for parsing is ', parsingtime)
        data['parsing time'].append(parsingtime)
        currenttime = time.perf_counter()
        # that ends right before t. the last
        if len(current_chunks_idx)>0:
            cg = learning_and_update(current_chunks_idx, previous_chunk_boundary_record, cg, t)
        learningtime = time.perf_counter() - currenttime
        print('elapsed time for learning and update is ', learningtime)
        data['learning time'].append(learningtime)
        data['n_chunk'].append(len(cg.chunks))
        if len(current_chunks_idx)>0:
            sizes = []
            for ck in cg.chunks:
                sz = ck.volume
                sizes.append(sz)
                if sz > maxchunksize: maxchunksize = sz
            data['chunk size'].append(np.mean(sizes))
        else:
            data['chunk size'].append(0)
        currenttime = time.perf_counter()
        # previous and current chunk
        cg.forget()
        np.save('performance_data.npy', data)

        self.fail()

    def test_adjacency_record(self, cg, chunk_record):
        '''transition record on adjacency should sum up to the length of sequence parsing record'''
        sum_adj = 0
        for ck in cg.chunks:
            for cr in cg.chunks[ck].adjacency:
                sum_adj = sum_adj + cg.chunks[ck].adjacency[cr][0]

        pre_adj_sum = 0
        for ck in cg.chunks:
            for cr in cg.chunks[ck].preadjacency:
                pre_adj_sum = pre_adj_sum + cg.chunks[ck].preadjacency[cr][0]

        print('adjacency summation = ', sum_adj)
        print('preadjacency summation = ', pre_adj_sum)
        print('length of chunk record is ', len(chunk_record))

        return

    def test_adjacency_preadjacency_match(self, cg):
        for cl in cg.chunks:
            for cr in cg.chunks[cl].adjacency:
                for dt in cg.chunks[cl].adjacency[cr]:
                    if cg.chunks[cl].adjacency[cr][dt]!=cg.chunks[cr].preadjacency[cl][dt]:
                        print('adjacency ', cg.chunks[cl].adjacency[cr][dt],'preadjacency: ', cg.chunks[cr].preadjacency[cl][dt])
        return

