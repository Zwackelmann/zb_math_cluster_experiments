from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from sklearn.feature_selection import chi2
import math
from sklearn.mixture import VBGMM
from chi2_util import dumpChiScores, chiSetGeq
import numpy as np
import joblib
from cluster_documents_util import sparseData2Matrix
import json
from sklearn.cluster import KMeans

corpusFilepath = "/home/simon/Projekte/zbMathClustering/raw_vector-small.json"
corpus = ArffJsonCorpus(corpusFilepath)


"""TDM = corpus.toCsrMatrix()
labelMatrix, classLabel2Number, classNumber2Label = initializeLabelMatrix(corpus)

dumpChiScores(TDM, labelMatrix, classNumber2Label)"""

"""chiSet = chiSetGeq(2000.0)
TDM, index2chiIndex = ArffJsonCorpus(corpusFilepath).toCsrMatrix(chiSet)

f = open("index2chiIndexForGmm.json", "w")
print >> f, json.dumps(index2chiIndex)
f.close()

gmm = VBGMM(n_components=63)
gmm.fit(TDM)
joblib.dump(gmm, "gmm63-allchi2geq2000")"""

print "00 - History: " + repr(readChiFile("chi-00", index2Word)[:10])
print "05 - Combinatorics: " + repr(readChiFile("chi-05", index2Word)[:10])
print "11 - Number Theory: " + repr(readChiFile("chi-11", index2Word)[:10])
print "35 - Partial differential equations: " + repr(readChiFile("chi-35", index2Word)[:10])
print "51 - Geometry: " + repr(readChiFile("chi-51", index2Word)[:10])
print "97 - Mathematics education: " + repr(readChiFile("chi-97", index2Word)[:10])

"""clModel = joblib.load("gmm63-allchi2geq2000")
log = open("clusters-gmm63-chi2geq2000", "w")
count = 0
for doc in corpus:
    npArray = sparseData2Matrix(doc.data, len(index2chiIndex), index2chiIndex)
    log.write(doc.id + ";" + str(clModel.predict(npArray)[0]) + "\n")
    count += 1
log.close()"""
