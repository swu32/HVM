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
                if sz > maxchunksize:maxchunksize = sz
            data['chunk size'].append(np.mean(sizes))
        else:
            data['chunk size'].append(0)
        currenttime = time.perf_counter()
        # previous and current chunk
        cg.forget()
        np.save('performance_data.npy', data)

        self.fail()
