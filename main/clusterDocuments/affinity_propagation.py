from sklearn.decomposition import TruncatedSVD
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
import numpy as np
from sklearn.cluster import AffinityPropagation

corpusFilepath = "/home/simon/Projekte/MIRS/testing_java_ml_libraries/raw_vector.json"
corpus = ArffJsonCorpus(corpusFilepath)
TDM = corpus.toCsrMatrix(shapeCols = 54334)

svd2 = joblib.load("lsi250-model")
LSI_TDM = svd2.transform(TDM)

ap = AffinityPropagation(
	damping=0.5, 
	max_iter=200, 
	convergence_iter=15, 
	copy=True, 
	preference=None, 
	affinity='euclidean', 
	verbose=False
)

ap.fit(LSI_TDM)
joblib.dump(ap, "ap-sklean_lsi250")
