import numpy as np
from sklearn.cluster import KMeans
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
from scipy.sparse import csr_matrix

corpusFilepath = "/home/simon/Projekte/MIRS/testing_java_ml_libraries/raw_vector.json"

def getTDM():
	global corpusFilepath
	return ArffJsonCorpus(corpusFilepath).toNumpyArray()

def fitKmeansModel(data):	
	km = KMeans(n_clusters=63, init='k-means++', max_iter=100, n_init=10)
	km.fit(data)
	return km

def fitGmmModel(data):
	g = mixture.GMM(n_components=63)
	g.fit(data)
	return g

def sparseData2Matrix(sparseData, numCols, translate = None):
	row = []
	col = []
	data = []

	for key, val in sparseData:
		if translate is None:
			row.append(0)
			col.append(key)
			data.append(val)
		elif key in translate:
			row.append(0)
			col.append(translate[key])
			data.append(val)

	return csr_matrix( (data,(row,col)), shape=(1, numCols) )

# km = fitKmeansModel(getTDM())
# joblib.dump(km, kmeansModelFile)

# g = fitGmmModel(getTDM())
# joblib.dump(g, gmmModelFile)

clModel = joblib.load("km63-sklean_lsi250")
corpus = ArffJsonCorpus(corpusFilepath)
lsiModel = joblib.load("lsi250-model")
numAttributes = corpus.header['num-attributes']

log = open("clusters-km63-sklean_lsi250", "w")
for doc in corpus:
	sparseVector = sparseData2Matrix(doc.data, numAttributes)
	lsiVector = lsiModel.transform(sparseVector)
	
	log.write(doc.id + ";" + str(clModel.predict(lsiVector)[0]) + "\n")

log.close()