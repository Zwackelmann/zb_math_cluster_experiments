import numpy as np
from sklearn.cluster import KMeans
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
from scipy.sparse import csr_matrix
from util import save_csr_matrix, load_csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

corpusFilepath = "/home/simon/Projekte/MIRS/testing_java_ml_libraries/raw_vector.json"

def getTDM():
	global corpusFilepath
	return ArffJsonCorpus(corpusFilepath).toNumpyArray()

def fitKmeansModel(data):	
	
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

"""TDM_full_text = load_csr_matrix("derived_data/zb_math_full_text_tdm.npz")
tfidf_trans = TfidfTransformer()
tfidf_trans.fit(TDM_full_text)

joblib.dump(tfidf_trans, "models/tfidf_full_text_model")
TDM_full_text_reweighted = tfidf_trans.transform(TDM_full_text)
km = KMeans(n_clusters=63, init='k-means++', max_iter=100, n_init=10)
km.fit(TDM_full_text_reweighted)

joblib.dump(km, "models/km63-full_text_tfidf")"""

# g = fitGmmModel(getTDM())
# joblib.dump(g, gmmModelFile)

clModel = joblib.load("models/ms-sklean_lsi250")
corpus = ArffJsonCorpus("raw_data/raw_vector.json")
lsi_model = joblib.load("models/lsi250-model")

log = open("results/clusters-ms-sklean_lsi250", "w")
for doc in corpus:
	sparseDoc = sparseData2Matrix(doc, 54334)
	arr = lsi_model.transform(sparseDoc)
	log.write(doc.id + ";" + str(clModel.predict(arr)[0]) + "\n")
	log.flush()

log.close()
