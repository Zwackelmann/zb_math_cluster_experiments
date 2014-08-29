import math
from sklearn.feature_selection import chi2
from os import listdir
from os.path import isfile, join

def readChiFile(chiFilepath):
	return map(lambda x: (int(x[0]), float(x[1]), float(x[2])), [line.split(";") for line in open(chiFilepath)])

def dumpChiScores(TDM, labelMatrix, classNumber2Label):
	bestTerms = getChiScores(TDM, labelMatrix, 1000)

	for i in range(0, len(classNumber2Label)):
		cl = classNumber2Label[i]
		bestTermsForClass = bestTerms[i]

		file = open("chi-" + str(cl), "w")
		for index, score, p in bestTermsForClass:
			file.write(str(index) + ";" + str(score) + ";" + str(p) + "\n")
		file.close()

def getChiScores(TDM, labelMatrix, firstN):
	bestTerms = []

	for i in range(0, len(labelMatrix)):
		v = chi2(TDM, labelMatrix[i])
		v = zip(range(0, len(v[0])), v[0], v[1])
		v = filter(lambda x: not(math.isnan(x[1])), v)

		bestTerms.append(sorted(v, key=lambda x: x[1], reverse=True)[:firstN])

	return bestTerms

def chiSetGeq(folder, threshold):
	chiSet = set()

	for chiFile in chiFiles:
		chis = map(lambda x: x[0], filter(lambda x: x[1] > threshold, readChiFile(chiFile)))
		chiSet.update(chis)

	return chiSet

def getChiFiles(folder):
	filenames, catLabels = zip(*[ (f, f[4:]) for f in listdir(folder) if isfile(join(folder,f)) and f[:4] == "chi-"])
	return filenames, catLabels

def getBestChiTerms(labelMatrix, TDM, index2WordMap, chiThreshold, classNumber2Label = None):
	bestTerms = getChiScores(TDM, labelMatrix, 10000)

	d = { }
	index = 0
	for bestTermsForCluster in bestTerms:
		bestTermsForCluster = filter(lambda x : x[1] > chiThreshold, bestTermsForCluster)

		if classNumber2Label is None:
			d[index] = map(lambda x: (index2WordMap[x[0]], x[1]), bestTermsForCluster)
		else:
			d[classNumber2Label[index]] = map(lambda x: (index2WordMap[x[0]], x[1]), bestTermsForCluster)
		index += 1

	return d