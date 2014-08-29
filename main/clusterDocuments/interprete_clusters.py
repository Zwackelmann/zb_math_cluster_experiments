from main.clusterDocuments.dim_reduction_util import initializeLabelMatrixFromClusters, initializeLabelMatrixFromCorpus
from main.clusterDocuments.chi2_util import dumpChiScores, getChiFiles, readChiFile, getChiScores, getBestChiTerms
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus
from util import get_index_to_word_map
import json
from scipy import stats

"""corpus = ArffJsonCorpus("raw_data/raw_vector.json")
labelMatrix = initializeLabelMatrixFromClusters("results/clusters-gmm-sklean_lsi250")
# labelMatrix, classLabel2Number, classNumber2Label = initializeLabelMatrixFromCorpus(corpus)
index2Word = get_index_to_word_map("../testing_java_ml_libraries/dict")

TDM = corpus.toCsrMatrix(shapeCols = 54334)

d = getBestChiTerms(
	labelMatrix = labelMatrix, 
	TDM = TDM, 
	index2WordMap = index2Word, 
	chiThreshold = 500,
	classNumber2Label = None
)

f = open("results/best_chis-gmm-sklean_lsi250", "w")
f.write(json.dumps(d))
f.close()"""


bestChisCats = json.load(open("results/best_chis-cats"))
bestChisKm63 = json.load(open("results/best_chis-gmm-sklean_lsi250"))

def rankCorrelation(chis1, chis2):
	chiTerms1 = set(map(lambda x : x[0], chis1))
	chiTerms2 = set(map(lambda x : x[0], chis2))

	in2ButNotIn1 = map(lambda x : (x, 0.0), list(chiTerms2.difference(chiTerms1)))
	in1ButNotIn2 = map(lambda x : (x, 0.0), list(chiTerms1.difference(chiTerms2)))

	chis1.extend(in2ButNotIn1)
	chis2.extend(in1ButNotIn2)

	tau, p = stats.kendalltau(chis1, chis2)
	return tau, p

for cat, bestChiTermsForCat in bestChisCats.items():
	for cluster, bestChiTermsForCluster in bestChisKm63.items():
		tau, p = rankCorrelation(bestChiTermsForCat, bestChiTermsForCluster)
		if tau > 0.4:
			print cat, cluster, tau


# tau, p = rankCorrelation(bestChisCats.items()[0][1], bestChisKm63.items()[0][1])
# print tau, p