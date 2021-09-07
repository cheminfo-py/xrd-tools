import pickle 

def read_pickle(filename): 
    with open(filename, 'rb') as handle: 
        return pickle.load(handle)
        