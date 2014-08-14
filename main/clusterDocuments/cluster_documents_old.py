import numpy as np
import matplotlib.pyplot as plt
import mlpy
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from sklearn.cluster import KMeans
from numpy.random import random 
from scipy.sparse import * 
from sklearn.cluster import KMeans
from string import printable
from sklearn.externals import joblib

corpusFilepath = "/home/simon/Projekte/zbMathClustering/lsi250-instances.json"

"""corpus = [
	[(0, 0.1), (2, 1.2), (3, 0.1)],
	[(0, 1.2), (1, 2.1), (4, 1.0)],
	[(1, 0.5), (2, 0.9), (3, 0.9)],
	[(0, 2.1), (2, 0.5), (4, 2.1)],
	[(0, 1.3), (1, 0.1), (3, 0.3)],
	[(1, 0.8), (3, 0.3), (4, 0.4)],
	[(0, 1.1), (1, 2.3), (2, 1.1)],
	[(1, 2.0), (2, 0.3), (4, 2.0)],
	[(0, 0.2), (2, 0.5), (3, 0.1)],
	[(0, 0.3), (1, 0.5), (2, 0.6)]
]"""

def arr2Matrix(arr, numCols):
	row = []
	col = []
	data = []

	for key, val in arr:
		row.append(0)
		col.append(key)
		data.append(val)

	return csr_matrix( (data,(row,col)), shape=(1, numCols) )

def applyKMeansForSparseData():
	row = []
	col = []
	data = []

	# maxDocs = 376900
	# maxDocs = 100
	c = iter(ArffJsonCorpus(corpusFilepath))

	keyTranslateMap = { }
	currentKey = 0

	numDocs = 0
	while True:
		try:
			doc = c.next()
			numDocs += 1
		except StopIteration:
			break

		for key, val in doc.data:
			row.append(numDocs-1)

			if not (key in keyTranslateMap):
				keyTranslateMap[key] = currentKey
				currentKey += 1

			col.append(keyTranslateMap[key])
			data.append(val)

	shapeRows = numDocs
	shapeCols = max(col)+1

	sparseMatrix = csr_matrix( (data,(row,col)), shape=(shapeRows, shapeCols) )

	km = KMeans(n_clusters=63, init='k-means++', max_iter=100, n_init=10)
	km.fit(sparseMatrix)

	return km

def getCorpusDimensions(corpusFilename):
	c = iter(ArffJsonCorpus(corpusFilename))

	numDocuments = 0
	numAttributes = None

	for doc in c:
		try:
			doc = c.next()
			numDocuments += 1
		except StopIteration:
			break

		if doc.storageType == ArffJsonDocument.dense:
			atts = len(doc.data)
		elif doc.storageType == ArffJsonDocument.sparse:
			atts = max(map(lambda d: d[0], doc.data)) + 1

		if numAttributes is None or numAttributes < atts:
			numAttributes = atts

	return numDocuments, numAttributes

def applyKMeansForDenseData():
	data = np.array(list(ArffJsonCorpus(corpusFilepath)))

	km = KMeans(n_clusters=63, init='k-means++', max_iter=100, n_init=10)
	km.fit(data)

	return km

km = applyKMeansForDenseData()
joblib.dump(km, 'kmeans-clustering.pkl')

#km = joblib.load('kmeans-clustering.pkl')

"""c = iter(ArffJsonCorpus(corpusFilepath))
for i in range(0, 100):
	doc = c.next()
	print(km.predict(doc)[0])"""

"""log = open("clusters-lsi250", "w")

c = iter(ArffJsonCorpus(corpusFilepath))
for i in range(0, numDocs):
	doc = c.next()
	translatedData = map(lambda x: (keyTranslateMap[x[0]], x[1]), doc.data)

	x = arr2Matrix(translatedData, shapeCols)
	log.write(filter(lambda c: c in printable, doc.id) + ";" + str(km.predict(x)[0]) + "\n")
"""
