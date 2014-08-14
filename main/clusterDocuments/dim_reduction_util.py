
def readDict(dictFilepath):
	lines = [line.strip() for line in open(dictFilepath)]
	index = 0
	index2Word = { }
	for line in lines:
		index2Word[index] = line
		index += 1
	return index2Word

def initializeLabelMatrix(corpus):
	classLabel2Number = { }
	classNumber2Label = { }
	currClassIndex = 0
	numDocs = 0
	for doc in iter(corpus):
		for cl in doc.classes:
			if not cl[:2] in classLabel2Number:
				classLabel2Number[cl[:2]] = currClassIndex
				classNumber2Label[currClassIndex] = cl[:2]
				currClassIndex += 1

		numDocs += 1

	numClasses = len(classLabel2Number)
	labelMatrix = map(lambda x : [0]*numDocs, range(0, numClasses)) 

	for docIndex, doc in zip(xrange(0, numDocs), iter(corpus)):
		for cl in doc.classes:
		 	labelMatrix[classLabel2Number[cl[:2]]][docIndex] = 1

	return labelMatrix, classLabel2Number, classNumber2Label
