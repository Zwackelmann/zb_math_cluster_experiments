import json
from scipy.sparse import csr_matrix
from util import save_csr_matrix, load_csr_matrix, get_dirpath, get_filenames_and_filepaths, DocumentParser
import numpy as np
from time import time
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from string import digits, ascii_letters
from os.path import isfile, join

dirpath = get_dirpath()
filenames, filepaths = get_filenames_and_filepaths("raw_data/ntcir_filenames")

tdm = load_csr_matrix("derived_data/zb_math_full_text_tdm2.npz")
translateMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
row_number2fulltext_id_map = json.load(open("derived_data/row_number2fulltext_id_map.json"))

phrase = "theorem"
tokenizer = DocumentParser.TextTokenizer()
tokens = tokenizer.tokenize(phrase)
tokenIds = map(lambda token: translateMap[token], tokens)

candidateIds = []
index = 0
start = time()

m = tdm[:,tokenIds]

candidateInd = [ ]
currInd = 0
for i in range(len(m.indptr)-1):
	diff = m.indptr[i+1] - m.indptr[i]
	if diff == len(tokenIds):
		candidateInd.append(currInd)
	currInd += 1

print str(len(candidateInd)) + " Candidates\n"

candidateIds = map(lambda i : row_number2fulltext_id_map[str(i)], candidateInd)
candidateDocumentFilenames = map(lambda id : filter(lambda c : c in ascii_letters or c in digits, id) + ".xml.npy", candidateIds) 
candidateDocumentFilepaths = map(lambda filename : join("derived_data/full_text_arrays", filename), candidateDocumentFilenames)

def hasTokenSequence(sent, tokens):
	matches = 0
	for token in sent:
		if token == tokens[matches]:
			matches += 1
		else:
			matches = 0
		
		if matches == len(tokens):
			return True

	return False

for filepath in candidateDocumentFilepaths:
	arr = np.load(filepath)
	
	for par in arr:
		for sent in par:
			if hasTokenSequence(sent, tokens):
				print repr(par) + "\n\n"
				break
				












