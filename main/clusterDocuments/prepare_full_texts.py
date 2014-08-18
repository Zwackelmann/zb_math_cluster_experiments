import sys
from os import path
from os import listdir
import json
from full_text_util import Document, DocumentParser
from scipy.sparse import csr_matrix
from util import save_csr_matrix, load_csr_matrix
import time
import uuid

if uuid.getnode() == 161338626918L: # is69
	dirpath = "/raid0/barthel/data/NTCIR_2014_enriched/"
elif uuid.getnode() == 622600420609L: # xmg-laptop
	dirpath = "/home/simon/samba/ifis/ifis/Datasets/math_challange/NTCIR_2014_enriched/"
else:
	raise ValueError("unknown node id " + str(uuid.getnode()))

filenamesFile = "raw_data/ntcir_filenames"

tmp = [ (line.strip(), dirpath + line.strip()) for line in open(filenamesFile) ]
filenames = map(lambda x: x[0], tmp)
filepaths = map(lambda x: x[1], tmp)

def buildWordCountDict(filepaths):
	p = DocumentParser()

	wordCounts = dict()
	count = 0
	total = len(filepaths)
	for filepath in filepaths:
		print str(count) + "/" + str(total)
		doc = p.parse(filepath)

		if "zbmath metadata" in doc.includedSources:
			for token in doc.tokens:
				if not token in wordCounts:
					wordCounts[token] = 0
				wordCounts[token] = wordCounts[token] + 1
		count += 1

	return wordCounts

def generateToken2IndexMap(wordCounts, minOccurrence):
	words = sorted(map(lambda x: x[0], filter(lambda x: x[1]>=minOccurrence, wordCounts.items())))
	return dict(zip(words, range(len(words))))

def dumpDocumentDataMaps(tokens2IndexMap, filenameFilepathsPairs, targetDir):
	p = DocumentParser()

	count = 0
	totalDocs = len(filenameFilepathsPairs)
	for filename, filepath in filenameFilepathsPairs:
		doc = p.parse(filepath)

		print str(count) + " / " + str(totalDocs)

		if "zbmath metadata" in doc.includedSources:
			dataMap = doc.toDataMap(tokens2IndexMap)

			f = open(path.join(targetDir, filename + ".json"), "w")
			f.write(json.dumps(dataMap))
			f.close()

		count += 1

def documentDataMaps2CsrMatrix(filepaths, numAttributes):
	row = []
	col = []
	data = []
	numDocs = 0

	for filepath in filepaths:
		numDocs += 1
		dataMap = json.load(open(filepath))

		for key, val in sorted(dataMap.items(), key=lambda x: int(x[0])):
		    row.append(numDocs-1)
		    col.append(key)
		    data.append(val)

	return csr_matrix( (data,(row,col)), shape=(numDocs, numAttributes) )

def documents2ArffJsonInstancesCorpus(filepaths, tokens2IndexMap):
	p = DocumentParser()

	f = open("raw_data/fulltext-corpus.json", "w")
	f.write("{\"relation-name\":\"full-text-corpus\",\"num-attributes\":" + str(len(tokens2IndexMap)) + "}\n")

	for filepath in filepaths:
		doc = p.parse(filepath)
		if "zbmath metadata" in doc.includedSources:
			f.write(doc.toArffJsonDocument(tokens2IndexMap) + "\n")
			f.flush()

	f.close()

"""
# save word count dict
wordCounts = buildWordCountDict(filepaths)
f = open("wordCountsFullTexts", "w")
f.write(json.dumps(wordCounts))
f.close()
"""

"""
# save token-to-index map
wordCounts = json.load(open("wordCountsFullTexts"))
tokens2IndexMap = generateToken2IndexMap(wordCounts, 5)
f = open("zb_math_full_texts_tokens2IndexMap", "w")
f.write(json.dumps(tokens2IndexMap))
f.close()
"""

"""
# dump intermediate full text data maps
tokens2IndexMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
dumpDocumentDataMaps(tokens2IndexMap, zip(filenames, filepaths), "full_text_term_value_maps")
"""

tokens2IndexMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
documents2ArffJsonInstancesCorpus(filepaths, tokens2IndexMap)

# transform intermediate full text data maps into a csr_matrix
"""tokens2IndexMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
filepaths = [ path.join("derived_data/full_text_term_value_maps", f) for f in listdir("derived_data/full_text_term_value_maps") if path.isfile(path.join("derived_data/full_text_term_value_maps", f)) ]
matrix = documentDataMaps2CsrMatrix(filepaths, len(tokens2IndexMap))
save_csr_matrix(matrix, "derived_data/zb_math_full_text_tdm")"""

"""
# load term-document-matrix
matrix = load_csr_matrix("zb_math_full_text_tdm.npz")
"""