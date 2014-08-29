from sklearn.decomposition import TruncatedSVD
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
import numpy as np
from sklearn.cluster import AffinityPropagation, MeanShift
import random

random.seed(0)

corpusFilepath = "/raid0/barthel/projects/zb_math_cluster_experiments/raw_data/raw_vector.json"
corpus = ArffJsonCorpus(corpusFilepath)
TDM = corpus.toCsrMatrix(shapeCols = 54334, selection = lambda doc: True if random.random() < 0.1 else False)
print "TDM shape: " + str(TDM.shape)

svd2 = joblib.load("models/lsi250-model")
LSI_TDM = svd2.transform(TDM)

#ap = AffinityPropagation(
#	damping=0.95, 
#	max_iter=200, 
#	convergence_iter=15, 
#	copy=True, 
#	preference=None, 
#	affinity='euclidean', 
#	verbose=False
#)

# ap.fit(LSI_TDM)

ms = MeanShift(
    bandwidth = None,
    seeds = None,
    bin_seeding = False,
    min_bin_freq = 1,
    cluster_all = True
)

ms.fit(LSI_TDM)

joblib.dump(ms, "models/ms-sklean_lsi250")
