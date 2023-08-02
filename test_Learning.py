from unittest import TestCase
from chunks import *


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

    def sample_concrete_content(self,varchunkname,cg):
        """Sample varchunk to arrive at a dataformat that is consistent with the input sequence
            seems to be working"""
        print(varchunkname)
        cg.sample_variable_instances(generative_model=False)# instantiate variables in cg into concrete instances

        ordered_content = cg.obtain_concrete_content_from_variables(varchunkname)
        print(ordered_content)

        # test in cgs with variables that represent other variables.
        # artificially create variables to add to cg,

        candidate_variables = set()
        VARS = list(cg.variables.values())
        candidate_variables.add(VARS[1])
        candidate_variables.add(VARS[3])
        candidate_variable_entailment = []
        candidate_variable_entailment.append(VARS[1].key)
        candidate_variable_entailment.append(VARS[3].key)

        v = Variable(candidate_variables)
        v = cg.add_variable(v, candidate_variable_entailment)
        v.chunk_probabilities[VARS[1].key] = 3
        v.chunk_probabilities[VARS[3].key] = 5
        cg.sample_variable_instances() # instantiate variables in cg into concrete instances
        ordered_content = cg.obtain_concrete_content_from_variables(v.key)
        #  I should not have done this, can a variable be referring to multiple variables?
        return ordered_content



    def test_get_concrete_content(self):
        pass
        return


    def test_parsing_meta_chunk_to_chunk(self, metachunk, seq):
        return


