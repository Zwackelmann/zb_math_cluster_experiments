from main.clusterDocuments.util import initializeLabelMatrixFromClusters, initializeLabelMatrixFromCorpus, getBestChiTerms, get_index_to_word_map, groupClusterDocuments, groupCorpusDocuments
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus
import json
from jaccardCoefficients import clusterStats

if __name__ == "__main__":
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

	importantWords = dict(map(lambda x : (x[0], ", ".join(map(lambda x: str(x[0]), x[1][:100]))), bestChisKm63.items()))

	category2Documents = groupCorpusDocuments("/home/simon/Projekte/MIRS/zb_math_cluster_experiments/raw_data/raw_vector.json", 1)
	cluster2Documents = groupClusterDocuments("results/clusters-gmm-sklean_lsi250")
	jaccForCl, jaccForCat, portForCl, portForCat, numDocsPerCl, numDocsPerCat = clusterStats(category2Documents, cluster2Documents)

	totalDocsInCl = sum(map(lambda x : int(x[1]), numDocsPerCl.items()))

	count = 0
	for cl in sorted(cluster2Documents.keys(), key=lambda cl: numDocsPerCl[cl], reverse=True):
		bestJacc = sorted(jaccForCl[cl], key=lambda x: x[1], reverse=True)[:5]
		bestPort = sorted(portForCl[cl], key=lambda x: x[1], reverse=True)[:5]

		print "Cluster: " + str(count) + ":"
		print "  Cluster size: " + str(numDocsPerCl[cl]) + " documents (" + str(round(float(numDocsPerCl[cl]*100)/totalDocsInCl, 2)) + "%)"
		print "  Important words: " + importantWords[cl]
		print "  Best jaccards: " + ", ".join(map(lambda x: "(" + str(x[0]) + ", " + str(round(x[1], 3)) + ")", bestJacc))
		print "  Best portions: " + ", ".join(map(lambda x: "(" + str(x[0]) + ", " + str(round(x[1], 3)) + ")", bestPort))
		print "\n"

		count += 1

	"""for cat, bestChiTermsForCat in bestChisCats.items():
		for cluster, bestChiTermsForCluster in bestChisKm63.items():
			tau, p = rankCorrelation(bestChiTermsForCat, bestChiTermsForCluster)
			if tau > 0.4:
				print cat, cluster, tau"""


	# tau, p = rankCorrelation(bestChisCats.items()[0][1], bestChisKm63.items()[0][1])
	# print tau, p