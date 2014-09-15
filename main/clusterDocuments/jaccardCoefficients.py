from main.arffJson.ArffJsonCorpus import ArffJsonCorpus
import json
from util import takeN, groupClusterDocuments, groupCorpusDocuments, printBestN

def clusterStats(category2Documents, cluster2Documents):
    categoryKeys = category2Documents.keys()
    clusterKeys = cluster2Documents.keys()

    jaccardForCluster = { }
    jaccardForCategory = { }
    portionForCluster = { }
    portionForCategory = { }
    numDocumentsPerCluster = { }
    numDocumentsPerCategory = { }

    for categoryKey in sorted(categoryKeys):
        categoryDocs = category2Documents[categoryKey]
        numCategoryDocs = len(categoryDocs)
        numDocumentsPerCategory[categoryKey] = numCategoryDocs

        for clusterKey in clusterKeys:
            clusterDocs = cluster2Documents[clusterKey]
            numClusterDocs = len(clusterDocs)
            numDocumentsPerCluster[clusterKey] = numClusterDocs

            numIntersect = len(clusterDocs.intersection(categoryDocs))
            numUnion = len(clusterDocs.union(categoryDocs))

            if not (clusterKey in jaccardForCluster):
                jaccardForCluster[clusterKey] = [ ]
            if not (categoryKey in jaccardForCategory):
                jaccardForCategory[categoryKey] = [ ]
            if not (clusterKey in portionForCluster):
                portionForCluster[clusterKey] = [ ]
            if not (categoryKey in portionForCategory):
                portionForCategory[categoryKey] = [ ]

            portionForCategory[categoryKey].append((clusterKey, float(numIntersect)/numCategoryDocs))
            portionForCluster[clusterKey].append((categoryKey, float(numIntersect)/numClusterDocs))
            jaccardForCluster[clusterKey].append((categoryKey, float(numIntersect)/numUnion))
            jaccardForCategory[categoryKey].append((clusterKey, float(numIntersect)/numUnion))

    return jaccardForCluster, jaccardForCategory, portionForCluster, portionForCategory, numDocumentsPerCluster, numDocumentsPerCategory


if __name__ == "__main__":
    category2Documents = groupCorpusDocuments("/home/simon/Projekte/MIRS/zb_math_cluster_experiments/raw_data/raw_vector.json", 1)
    cluster2Documents = groupClusterDocuments("results/clusters-gmm-sklean_lsi250")

    jaccForCl, jaccForCat, portForCl, portForCat, numDocsPerCl, numDocsPerCat = clusterStats(category2Documents, cluster2Documents)
    
    totalDocsInClu = sum(map(lambda x: x[1], numDocsPerCl.items()))
    totalDocsInCats = sum(map(lambda x: x[1], numDocsPerCat.items()))

    print "Jaccard coeff. by Cluster"
    print printBestN(jaccForCl, 5)
    print "\n\nJaccard coeff. by Category"
    print printBestN(jaccForCat, 5)
    print "\n\nPortions by Cluster"
    print printBestN(portForCl, 5)
    print "\n\nPortions by Category"
    print printBestN(portForCat, 5)

    print "\n\nNum Docs per Cluster (total: " + str(totalDocsInClu) + ")"
    print map(lambda x : (x[0], round(float(x[1]) / totalDocsInClu, 4)), sorted(numDocsPerCl.items(), key=lambda x: x[1], reverse=True))
    print "\n\nNum Docs per Category (total: " + str(totalDocsInCats) + ")"
    print map(lambda x : (x[0], round(float(x[1]) / totalDocsInCats, 4)), sorted(numDocsPerCat.items(), key=lambda x: x[1], reverse=True))