from sklearn.decomposition import TruncatedSVD
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
import numpy as np
from sklearn.cluster import KMeans

corpusFilepath = "/home/simon/Projekte/zbMathClustering/raw_vector.json"
corpus = ArffJsonCorpus(corpusFilepath)
TDM = corpus.toCsrMatrix(shapeCols = 54334)

"""svd = TruncatedSVD(n_components=250)
svd.fit(TDM)
joblib.dump(svd, "lsi250-model")"""

svd2 = joblib.load("lsi250-model")
LSI_TDM = svd2.transform(TDM)

km = KMeans(n_clusters=63, init='k-means++', max_iter=100, n_init=10)
km.fit(LSI_TDM)
joblib.dump(km, "km63-sklean_lsi250")

"""clModel = joblib.load("km63-sklean_lsi250")
# log = open("clusters-km63-sklearn_lsi250", "w")
log = open("foo", "w")
count = 0
for arr in LSI_TDM:
    # npArray = sparseData2Matrix(doc.data, len(index2chiIndex), index2chiIndex)
    log.write(doc.id + ";" + str(clModel.predict(npArray)[0]) + "\n")
    count += 1
log.close()"""
