import numpy as np
from scipy.sparse import csr_matrix
import uuid

def save_csr_matrix(array, filename):
	np.savez(filename, data=array.data, indices=array.indices,
         indptr=array.indptr, shape=array.shape)

def load_csr_matrix(filename):
    loader = np.load(filename)
    return csr_matrix( (loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])

def get_dirpath():
	if uuid.getnode() == 161338626918L: # is69
		dirpath = "/raid0/barthel/data/NTCIR_2014_enriched/"
	elif uuid.getnode() == 622600420609L: # xmg-laptop
		dirpath = "/home/simon/samba/ifis/ifis/Datasets/math_challange/NTCIR_2014_enriched/"
	else:
		raise ValueError("unknown node id " + str(uuid.getnode()))

	return dirpath

def get_filenames_and_filepaths(file):
	dirpath = get_dirpath()
	
	tmp = [ (line.strip(), dirpath + line.strip()) for line in open(file) ]
	filenames = map(lambda x: x[0], tmp)
	filepaths = map(lambda x: x[1], tmp)

	return filenames, filepaths