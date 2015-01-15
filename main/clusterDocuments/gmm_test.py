from sklearn.decomposition import TruncatedSVD
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from util import build_csr_matrix

corpus = ArffJsonCorpus("raw_data/raw_vector.json")
TDM = corpus.toCsrMatrix(shapeCols = 54334)

# build lsi model
"""lsi_model = TruncatedSVD(n_components=250)
lsi_model.fit(TDM)
joblib.dump(lsi_model, "models/raw_vector-lsi250_model")"""

# build gmm model
"""lsi_model = joblib.load("models/raw_vector-lsi250_model")
gmm_model = GMM(n_components=64)
gmm_model.fit(lsi_model.transform(TDM))
joblib.dump(gmm_model, "models/gmm-raw_vector-lsi250")"""

#cluster documents
f = open("results/clusters-gmm-raw_vector-lsi250", "w")
lsi_model = joblib.load("models/raw_vector-lsi250_model")
cl_model = joblib.load("models/gmm-raw_vector-lsi250")

count = 0
for doc in corpus:
    doc_vector = lsi_model.transform(build_csr_matrix(dict(doc.data), numAttributes=54334))
    f.write(doc.id + ";" + str(cl_model.predict(doc_vector)[0]) + "\n")
    count += 1
f.close()