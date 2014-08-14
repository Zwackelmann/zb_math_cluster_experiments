from main.arffJson.ArffJsonCorpus import ArffJsonCorpus
import json

def takeN(iter, n):
	for i in xrange(n):
		yield iter.next()

clusterFile = open("clusters-km63-sklean_lsi250", "r")
cluster2Documents = { }

for line in clusterFile:
	x = line.split(";")
	docId = x[0]
	docClass = x[1].strip()

	if not (docClass in cluster2Documents):
		cluster2Documents[docClass] = set()

	cluster2Documents[docClass].add(docId)

corpusFilepath = "/home/simon/Projekte/MIRS/testing_java_ml_libraries/raw_vector.json"
# mode=1: only main classes
# mode=2: also secondary classes
mode = 1

consideredDocs = list(takeN(iter(ArffJsonCorpus(corpusFilepath)), 556296))
print len(consideredDocs)

category2Documents = { }
for doc in consideredDocs:
	if len(doc.classes) != 0:
		docId = doc.id

		if mode == 1:
			docMainClass = doc.classes[0][0:2]

			if not (docMainClass in category2Documents):
				category2Documents[docMainClass] = set()

			category2Documents[docMainClass].add(docId)
		elif mode == 2:
			for docClass in doc.classes:
				topClass = docClass[0:2]
				
				if not (topClass in category2Documents):
					category2Documents[topClass] = set()

				category2Documents[topClass].add(docId)

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

#	print categoryKey + ": " + repr(sorted(jaccardForCluster, key=lambda x: x[1], reverse=True))
#	print categoryKey + ": " + repr(sorted(portionForCluster, key=lambda x: x[1], reverse=True))

def printBestN(dict):
	bestN = [(int(k), sorted(v, key=lambda x: x[1], reverse=True)[0:5]) for k, v in dict.items()]
	sortedBestN = sorted(bestN, key=lambda x: int(x[0]))
	return "\n".join(map(lambda x: repr(x), sortedBestN))

print "Jaccard coeff. by Cluster"
print printBestN(jaccardForCluster)
print "\n\nJaccard coeff. by Category"
print printBestN(jaccardForCategory)
print "\n\nPortions by Cluster"
print printBestN(portionForCluster)
print "\n\nPortions by Category"
print printBestN(portionForCategory)

print "\n\nNum Docs per Cluster"
print sorted(numDocumentsPerCluster.items(), key=lambda x: int(x[1]), reverse=True)
print "\n\nNum Docs per Category"
print sorted(numDocumentsPerCategory.items(), key=lambda x: int(x[1]), reverse=True)


# print sorted([(int(k), sum(map(lambda x: x[1], v))) for k, v in portionForCategory.items()], key=lambda x: int(x[0])) 

