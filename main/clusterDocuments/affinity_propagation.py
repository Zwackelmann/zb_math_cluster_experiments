from sklearn.decomposition import TruncatedSVD
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
import joblib
import numpy as np
<<<<<<< HEAD
from sklearn.cluster import AffinityPropagation, MeanShift
=======
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GMM
>>>>>>> e2a51c5144409b3c12e0c91f770319673b7f5a47
import random

random.seed(0)

corpusFilepath = "raw_data/raw_vector.json"
corpus = ArffJsonCorpus(corpusFilepath)
TDM = corpus.toCsrMatrix(shapeCols = 54334, selection = lambda doc: True if random.random() < 0.1 else False)
print "TDM shape: " + str(TDM.shape)

svd2 = joblib.load("models/lsi250-model")
LSI_TDM = svd2.transform(TDM)

<<<<<<< HEAD
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
=======
"""ap = AffinityPropagation(
	damping=0.85, 
	max_iter=200, 
	convergence_iter=15, 
	copy=True, 
	preference=None, 
	affinity='euclidean', 
	verbose=False
)

ap.fit(LSI_TDM)
joblib.dump(ap, "models/ap-damp085-sklean_lsi250")"""

gmm = GMM(
	n_components=63, 
	covariance_type='diag', 
	random_state=None, 
	thresh=0.01, 
	min_covar=0.001, 
	n_iter=100, 
	n_init=5, 
	params='wmc', 
	init_params='wmc'
)

gmm.fit(LSI_TDM)
joblib.dump(gmm, "models/gmm-sklean_lsi250")

>>>>>>> e2a51c5144409b3c12e0c91f770319673b7f5a47
