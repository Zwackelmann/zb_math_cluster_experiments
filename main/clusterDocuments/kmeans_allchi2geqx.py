from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from sklearn.feature_selection import chi2
import math
from sklearn.cluster import KMeans
from chi2_util import dumpChiScores, chiSetGeq
import numpy as np
import joblib
from cluster_documents import sparseData2Matrix
import json

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

index2chiIndex = dict(map(lambda x: (int(x[0]), x[1]), json.load(open("index2chiIndex.json")).items()))

"""
print "00 - History: " + repr(readChiFile("chi-00", index2Word)[:10])
print "05 - Combinatorics: " + repr(readChiFile("chi-05", index2Word)[:10])
print "11 - Number Theory: " + repr(readChiFile("chi-11", index2Word)[:10])
print "35 - Partial differential equations: " + repr(readChiFile("chi-35", index2Word)[:10])
print "51 - Geometry: " + repr(readChiFile("chi-51", index2Word)[:10])
print "97 - Mathematics education: " + repr(readChiFile("chi-97", index2Word)[:10])"""

clModel = joblib.load("km63-allchi2geq2000")
log = open("clusters-km63-chi2geq2000", "w")
count = 0
for doc in corpus:
	npArray = sparseData2Matrix(doc.data, len(index2chiIndex), index2chiIndex)
	log.write(doc.id + ";" + str(clModel.predict(npArray)[0]) + "\n")
	count += 1
log.close()