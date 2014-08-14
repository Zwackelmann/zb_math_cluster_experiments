import math
from sklearn.feature_selection import chi2

def readChiFile(chiFilepath):
	return map(lambda x: (int(x[0]), float(x[1]), float(x[2])), [line.split(";") for line in open(chiFilepath)])

def dumpChiScores(TDM, labelMatrix, classNumber2Label):
	for i in range(0, len(classNumber2Label)):
		cl = classNumber2Label[i]
		print "chi2 for cat: " + str(cl) + ":"
		v = chi2(TDM, labelMatrix[i])
		v = zip(range(0, len(v[0])), v[0], v[1])
		v = filter(lambda x: not(math.isnan(x[1])), v)

		bestTerms = sorted(v, key=lambda x: x[1], reverse=True)[:1000]

		file = open("chi-" + str(cl), "w")
		for index, score, p in bestTerms:
			file.write(str(index) + ";" + str(score) + ";" + str(p) + "\n")
		file.close()

chiFiles = [
	"chi-00", "chi-08", "chi-15", "chi-20", "chi-31", "chi-37", "chi-43", "chi-49", 
	"chi-55", "chi-65", "chi-78", "chi-85", "chi-93", "chi-01", "chi-11", "chi-16", 
	"chi-22", "chi-32", "chi-39", "chi-44", "chi-51", "chi-57", "chi-68", "chi-80", 
	"chi-86", "chi-94", "chi-03", "chi-12", "chi-17", "chi-26", "chi-33", "chi-40",
	"chi-45", "chi-52", "chi-58", "chi-70", "chi-81", "chi-90", "chi-97", "chi-05", 
	"chi-13", "chi-18", "chi-28", "chi-34", "chi-41", "chi-46", "chi-53", "chi-60", 
	"chi-74", "chi-82", "chi-91", "chi-06", "chi-14", "chi-19", "chi-30", "chi-35",
	"chi-42", "chi-47", "chi-54", "chi-62", "chi-76", "chi-83", "chi-92"
]

def chiSetGeq(threshold):
	chiSet = set()

	for chiFile in chiFiles:
		chis = map(lambda x: x[0], filter(lambda x: x[1] > threshold, readChiFile(chiFile)))
		chiSet.update(chis)

	return chiSet