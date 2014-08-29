from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from sklearn.feature_selection import chi2
import math
from sklearn.cluster import KMeans
from chi2_util import dumpChiScores, chiSetGeq, readChiFile
import numpy as np
import joblib
import json
from util import get_index_to_word_map

corpusFilepath = "/home/simon/Projekte/MIRS/testing_java_ml_libraries/raw_vector.json"
corpus = ArffJsonCorpus(corpusFilepath)

"""TDM = corpus.toCsrMatrix()
labelMatrix, classLabel2Number, classNumber2Label = initializeLabelMatrix(corpus)

dumpChiScores(TDM, labelMatrix, classNumber2Label)"""

"""chiSet = chiSetGeq(2000.0)
TDM, index2chiIndex = ArffJsonCorpus(corpusFilepath).toCsrMatrix(chiSet)

f = open("index2chiIndex.json", "w")
print >> f, json.dumps(index2chiIndex)
f.close()

km = KMeans(n_clusters=63, init='k-means++', max_iter=100, n_init=10)
km.fit(TDM)
joblib.dump(km, "km63-allchi2geq2000")"""

# index2chiIndex = dict(map(lambda x: (int(x[0]), x[1]), json.load(open("derived_data/index2chiIndex.json")).items()))
index2Word = get_index_to_word_map("../testing_java_ml_libraries/dict")


print "00 - History: " + repr(map(lambda x: index2Word[x[0]], readChiFile("derived_data/chi/chi-00")[:10]))
print "05 - Combinatorics: " + repr(map(lambda x: index2Word[x[0]], readChiFile("derived_data/chi/chi-05")[:10]))
print "11 - Number Theory: " + repr(map(lambda x: index2Word[x[0]], readChiFile("derived_data/chi/chi-11")[:10]))
print "35 - Partial differential equations: " + repr(map(lambda x: index2Word[x[0]], readChiFile("derived_data/chi/chi-35")[:10]))
print "51 - Geometry: " + repr(map(lambda x: index2Word[x[0]], readChiFile("derived_data/chi/chi-51")[:10]))
print "97 - Mathematics education: " + repr(map(lambda x: index2Word[x[0]], readChiFile("derived_data/chi/chi-97")[:10]))

"""clModel = joblib.load("km63-allchi2geq2000")
log = open("clusters-km63-chi2geq2000", "w")
count = 0
for doc in corpus:
	npArray = sparseData2Matrix(doc.data, len(index2chiIndex), index2chiIndex)
	log.write(doc.id + ";" + str(clModel.predict(npArray)[0]) + "\n")
	count += 1
log.close()"""