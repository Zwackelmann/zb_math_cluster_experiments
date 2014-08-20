import json
from scipy.sparse import csr_matrix
from util import save_csr_matrix, load_csr_matrix, get_dirpath, get_filenames_and_filepaths
import numpy as np
from full_text_util import DocumentParser
from time import time
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument

dirpath = get_dirpath()
filenames, filepaths = get_filenames_and_filepaths("raw_data/ntcir_filenames")

tdm = load_csr_matrix("derived_data/zb_math_full_text_tdm2.npz")
translateMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
row_number2fulltext_id_map = json.load(open("derived_data/row_number2fulltext_id_map.json"))

phrase = "our task is now to"
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

print map(lambda i : row_number2fulltext_id_map[str(i)], candidateInd)


"""corpus = ArffJsonCorpus("raw_data/fulltext-corpus.json")
m = { }
count = 0
for doc in corpus:
	m[count] = doc.id
	count += 1

f = open("derived_data/row_number2fulltext_id_map.json", "w")
f.write(json.dumps(m))
f.close()"""