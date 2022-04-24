# code to record time
data = {}
data['parsing time'] = []
data['learning time'] = []
data['n_chunk'] = []
data['chunk size'] = []
currenttime = time.perf_counter()
parsingtime = time.perf_counter() - currenttime
data['parsing time'].append(parsingtime)
currenttime = time.perf_counter()
learningtime = time.perf_counter() - currenttime
data, maxchunksize = save_diagnostic_data(data, learningtime, cg, current_chunks_idx, maxchunksize)
print('Buffer sequence length ', Buffer.seql, 'maxchunksize ', maxchunksize)
