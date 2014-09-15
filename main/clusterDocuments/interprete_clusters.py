from main.clusterDocuments.util import initializeLabelMatrixFromClusters, initializeLabelMatrixFromCorpus, getBestChiTerms, get_index_to_word_map, groupClusterDocuments, groupCorpusDocuments, wordListRankCorrelation
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus
import json
from jaccardCoefficients import clusterStats
import matplotlib.pyplot as plt
from util import hist, barPlot

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

    # calc and dump rankCorrelationMatrix
    bestChisCats = json.load(open("results/best_chis-cats"))
    bestChisClu = json.load(open("results/best_chis-gmm-sklean_lsi250"))

    """rankCorrelationMatrix = { }
    for cat, bestChiTermsForCat in bestChisCats.items():
        for clu, bestChiTermsForCluster in bestChisClu.items():
            tau, p = wordListRankCorrelation(bestChiTermsForCat, bestChiTermsForCluster)
            if not clu in rankCorrelationMatrix:
                rankCorrelationMatrix[clu] = { }
            rankCorrelationMatrix[clu][cat] = [tau, p]

    f = open("derived_data/rankCorrelationMatrix.json", "w")
    f.write(json.dumps(rankCorrelationMatrix))
    f.close()"""
    
    """rankCorrelationMatrix = json.load(open("derived_data/rankCorrelationMatrix.json"))
    bestRankedCatsForCluster = { }
    for clu, cats in rankCorrelationMatrix.items():
        sortedByRankCorr = sorted(cats.items(), key = lambda x : x[1][0], reverse=True)
        bestCats = map(lambda x : (x[0], x[1][0]), sortedByRankCorr[0:3])
        bestRankedCatsForCluster[clu] = bestCats

    importantWords = dict(map(lambda x : (x[0], ", ".join(map(lambda x: str(x[0]), x[1][:100]))), bestChisClu.items()))"""

    category2Documents = groupCorpusDocuments("/home/simon/Projekte/MIRS/zb_math_cluster_experiments/raw_data/raw_vector.json", 1)
    cluster2Documents = groupClusterDocuments("results/clusters-gmm-sklean_lsi250")
    jaccForClu, jaccForCat, portForClu, portForCat, numDocsPerClu, numDocsPerCat = clusterStats(category2Documents, cluster2Documents)

    totalDocsInClu = sum(map(lambda x : int(x[1]), numDocsPerClu.items()))
    totalDocsInCat = sum(map(lambda x : int(x[1]), numDocsPerCat.items()))

    portionsInClu = map(lambda x : (x[1]*100)/totalDocsInClu, numDocsPerClu.items())
    portionsInCat = map(lambda x : (x[1]*100)/totalDocsInCat, numDocsPerCat.items())

    # plot distributions
    plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    maxJaccs = map(lambda v: max(v), map(lambda kv: map(lambda x: x[1], kv[1]), jaccForCat.items()))
    jaccLabels, jaccValues = hist(maxJaccs, [0.0, 0.05, 0.1, 0.15, 0.20, 1.0])
    # labelsClu, valuesClu = hist(portionsInClu, [0, 1, 2, 3, 4, 5, 100])
    rects = barPlot(plt=plt, labels=["<5%", "5-10%", "10-15%", "15-20%", ">20%"], valueLists=[jaccValues], width=0.5)
    ax = plt.axes()
    ax.legend( map(lambda r : r[0], rects), ['Best Jaccard Coeff.'], loc=1)
    plt.tight_layout()
    plt.show()

    # print ", ".join(map(lambda x: str(x[0]), bestChisCats["97"][:100]))

    """count = 0
    for clu in sorted(cluster2Documents.keys(), key=lambda clu: numDocsPerClu[clu], reverse=True):
        bestJacc = sorted(jaccForClu[clu], key=lambda x: x[1], reverse=True)[:5]
        bestPort = sorted(portForClu[clu], key=lambda x: x[1], reverse=True)[:5]

        print "Cluster: " + str(count) + ":"
        print "  Cluster size: " + str(numDocsPerClu[clu]) + " documents (" + str(round(float(numDocsPerClu[clu]*100)/totalDocsInCl, 2)) + "%)"
        print "  Important words: " + importantWords[clu]
        print "  Best jaccards: " + ", ".join(map(lambda x: "(" + str(x[0]) + ", " + str(round(x[1], 3)) + ")", bestJacc))
        print "  Best portions: " + ", ".join(map(lambda x: "(" + str(x[0]) + ", " + str(round(x[1], 3)) + ")", bestPort))
        print "  Best rank correlations: " + ", ".join(map(lambda x: str(x[0]) + "(" + str(round(x[1], 2)) + ")", bestRankedCatsForCluster[clu]))
        print "\n"

        count += 1"""

    
