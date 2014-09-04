import numpy as np
from scipy.sparse import csr_matrix
import uuid
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from scipy import stats
import math
from sklearn.feature_selection import chi2
from os import listdir
from os.path import isfile, join
import xml.sax
from string import ascii_letters, digits
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import nltk
import re

# general 
def get_dirpath():
	if uuid.getnode() == 161338626918L: # is69
		dirpath = "/raid0/barthel/data/NTCIR_2014_enriched/"
	elif uuid.getnode() == 622600420609L: # xmg-laptop
		dirpath = "/home/simon/samba/ifis/ifis/Datasets/math_challange/NTCIR_2014_enriched/"
	else:
		raise ValueError("unknown node id " + str(uuid.getnode()))

	return dirpath

def get_filenames_and_filepaths(file):
	dirpath = get_dirpath()
	
	tmp = [ (line.strip(), dirpath + line.strip()) for line in open(file) ]
	filenames = map(lambda x: x[0], tmp)
	filepaths = map(lambda x: x[1], tmp)

	return filenames, filepaths

def save_csr_matrix(array, filename):
	np.savez(filename, data=array.data, indices=array.indices,
         indptr=array.indptr, shape=array.shape)

def load_csr_matrix(filename):
    loader = np.load(filename)
    return csr_matrix( (loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])

def get_index_to_word_map(file):
	count = 0
	d = { }
	for line in open(file):
		d[count] = line.strip()
		count += 1

	return d

def filesInDict(path, withPath = False):
	filenames = [ f for f in listdir(folder) if isfile(join(folder, f))]

	if withPath:
		return map(lambda filename : join(path, filename), filenames)
	else:
		return filenames

# utility
def takeN(iter, n):
	for i in xrange(n):
		yield iter.next()

def printBestN(dict, n):
	bestN = [(int(k), sorted(v, key=lambda x: x[1], reverse=True)[:n]) for k, v in dict.items()]
	sortedBestN = sorted(bestN, key=lambda x: int(x[0]))
	return "\n".join(map(lambda x: repr(x), sortedBestN))
	
def rankCorrelation(chis1, chis2):
	chiTerms1 = set(map(lambda x : x[0], chis1))
	chiTerms2 = set(map(lambda x : x[0], chis2))

	in2ButNotIn1 = map(lambda x : (x, 0.0), list(chiTerms2.difference(chiTerms1)))
	in1ButNotIn2 = map(lambda x : (x, 0.0), list(chiTerms1.difference(chiTerms2)))

	chis1.extend(in2ButNotIn1)
	chis2.extend(in1ButNotIn2)

	tau, p = stats.kendalltau(chis1, chis2)
	return tau, p

# reading and grouping
def readDict(dictFilepath):
	lines = [line.strip() for line in open(dictFilepath)]
	index = 0
	index2Word = { }
	for line in lines:
		index2Word[index] = line
		index += 1
	return index2Word

def initializeLabelMatrixFromCorpus(corpus):
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

def initializeLabelMatrixFromClusters(clusterFile):
	numDocs = 0
	clusters = set()

	for line in open(clusterFile):
		x = line.split(";")
		docCluster = int(x[1].strip())
		clusters.add(docCluster)
		numDocs += 1

	numClusters = len(clusters)
	labelMatrix = map(lambda x : [0]*numDocs, range(0, numClusters)) 

	docIndex = 0
	for line in open(clusterFile):
		x = line.split(";")
		docCluster = int(x[1].strip())

		labelMatrix[int(docCluster)][docIndex] = 1
		docIndex += 1

	return labelMatrix

def groupClusterDocuments(clusterFile):
	cluster2Documents = { }

	for line in open(clusterFile):
		x = line.split(";")
		docId = x[0]
		docClass = x[1].strip()

		if not (docClass in cluster2Documents):
			cluster2Documents[docClass] = set()

		cluster2Documents[docClass].add(docId)

	return cluster2Documents

def groupCorpusDocuments(corpusFilepath, mode):
	"""
	mode=1: only main classes
	mode=2: also secondary classes
	"""
	category2Documents = { }
	for doc in ArffJsonCorpus(corpusFilepath):
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

	return category2Documents

# Chi square
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

# Full text stuff
class Document:
	def __init__(self, identifiers = [], title = None, 
			abstract = None, languages = [],
			includedSources = [], 
			publicationDate = None, zbMscCats = [], 
		    arxivCats = [], authors = [], 
		    fullTextTokens = []
	):
		self.identifiers = identifiers
		self.title = title
		self.abstract = abstract
		self.languages = languages
		self.includedSources = includedSources
		self.publicationDate = publicationDate
		self.zbMscCats = zbMscCats
		self.arxivCats = arxivCats
		self.authors = authors
		self.fullTextTokens = fullTextTokens

	def toDataMap(self, token2IndexMap):
		dataMap = dict()
		for t in self.fullTextTokens:
			if t in token2IndexMap:
				i = token2IndexMap[t]
				if not i in dataMap:
					dataMap[i] = 0

				dataMap[i] = dataMap[i] + 1

		return dataMap

	def toArffJsonDocument(self, token2IndexMap):
		strBuffer = []
		strBuffer.append("[[\"")
		
		ids = filter(lambda x: x.source=="arxiv", self.identifiers)

		if len(ids) != 1:
			if len(ids) > 1:
				raise ValueError("Found multiple arxiv ids")
			else:
				raise ValueError("Didn't found any arxiv id")
		else:
			strBuffer.append(ids[0].ident)
		
		strBuffer.append("\",[")
		strBuffer.append(",".join(map(lambda cl: "\"" + cl + "\"", self.zbMscCats)))
		strBuffer.append("]],{")

		dataMap = self.toDataMap(token2IndexMap)
		sortedKeys = sorted(dataMap.items(), key=lambda x: int(x[0]))
		strBuffer.append(",".join(map(lambda kv : "\"" + str(kv[0]) + "\":" + str(kv[1]), sortedKeys)))

		strBuffer.append("}]")

		return "".join(strBuffer)

	class Identifier:
		def __init__(self, ident, source):
			self.ident = ident
			self.source = source

		def __str__(self):
			return "Identifier(source=" + self.source + ", ident=" + self.ident + ")"

		def __repr__(self):
			return str(self)

	class Author:
		def __init__(self, name, ident = None):
			self.name = name
			self.ident = ident

		def __str__(self):
			return "Author(" + self.name + ("(" + self.ident + ")" if not self.ident==None else "") + ")"
		
		def __repr__(self):
			return str(self)

class DocumentParser:
	def __init__(self):
		self.textTokenizer = DocumentParser.TextTokenizer()
		self.formulaTokenizer = DocumentParser.FormulaTokenizer()#
		self.sentenceDetector = nltk.data.load('tokenizers/punkt/english.pickle')

	def parse(self, filepath):
		source = open(filepath)
		rawDocument = DocumentParser.RawDocument()
		ch = DocumentParser.ZbMathContentHandler(rawDocument)
		xml.sax.parse(source, ch)

		tokens = []
		for content in rawDocument.rawContent:
			if type(content) is DocumentParser.RawDocument.TextContent:
				tokens.extend(self.textTokenizer.tokenize(content.content))
			elif type(content) is DocumentParser.RawDocument.FormulaContent:
				tokens.extend(self.formulaTokenizer.tokenize(content.content))
			elif type(content) is DocumentParser.RawDocument.Paragraph:
				pass
			else:
				raise ValueError(str(type(content)) + " is not supported")

		rawDocument.rawContent = tokens
		return rawDocument.toDocument()

	def parseWithParagraphStructure(self, filepath):
		source = open(filepath)
		rawDocument = DocumentParser.RawDocument()

		ch = DocumentParser.ZbMathContentHandler(rawDocument)
		xml.sax.parse(source, ch)

		paragraphs = []
		paragraphBuffer = []
		formulaCount = 0
		formulaDict = { }
		for content in rawDocument.rawContent:
			if type(content) is DocumentParser.RawDocument.TextContent:
				paragraphBuffer.append(content.content)
			elif type(content) is DocumentParser.RawDocument.FormulaContent:
				paragraphBuffer.append("$" + str(formulaCount) + "$")
				formulaDict[formulaCount] = content.content
				formulaCount += 1
			elif type(content) is DocumentParser.RawDocument.Paragraph:
				paragraphString = " ".join(paragraphBuffer)
				paragraphBuffer = []
				sentences = self.sentenceDetector.tokenize(paragraphString)

				paragraphs.append(map(lambda s : self.tokenizeSentence(s, formulaDict), sentences))
			else:
				raise ValueError(str(type(content)) + " is not supported")

		return paragraphs

	def tokenizeSentence(self, sentence, formulaDict):
		tokens = []

		while len(sentence) != 0:
			res = re.search(r"\$\d+\$", sentence)
			if res is None:
				tokens.extend(self.textTokenizer.tokenize(sentence))
				break
			else:
				tokens.extend(self.textTokenizer.tokenize(sentence[:res.start()]))
                                formula = formulaDict.get(int(sentence[res.start()+1:res.end()-1]))
                                if not formula is None:
				    tokens.append("$" + formula  + "$")
				sentence = sentence[res.end():]
		return tokens

	class RawDocument:
		def __init__(self):
			self.includedSources = []
			self.identifiers = []
			self.title = None
			self.abstract = None
			self.languages = []
			self.rawPublicationDate = None
			self.zbMscCats = []
			self.arxivCats = []
			self.plainAuthors = []
			self.authorIdentifiers = []
			self.rawContent = []

		class FormulaContent(object):
			def __init__(self, content):
				self.content = content

		class TextContent(object):
			def __init__(self, content):
				self.content = content

		class Paragraph(object):
			def __init__(self, ident):
				self.ident = ident

		def toDocument(self):
			authors = []
			if len(self.authorIdentifiers) == len(self.plainAuthors):
				for name, ident in zip(self.plainAuthors, self.authorIdentifiers):
					authors.append(Document.Author(name=name, ident=ident))
			else:
				for name in self.plainAuthors:
					authors.append(Document.Author(name=name, ident=None))

			identifiers = []
			for ident in self.identifiers:
				identifiers.append(Document.Identifier(ident=ident['id'], source=ident['type']))

			parsedTime = None
			if not self.rawPublicationDate is None:
				parsedTime = time.strptime(self.rawPublicationDate[:10], "%Y-%m-%d")

			return Document(
				includedSources = self.includedSources,
				identifiers = identifiers,
				title = self.title,
				abstract = self.abstract,
				languages = self.languages,
				publicationDate = parsedTime,
				zbMscCats = self.zbMscCats,
				arxivCats = self.arxivCats,
				authors = authors,
				fullTextTokens = self.rawContent
			)

	RawDocument.dateFormat = ""

	class ZbMathContentHandler(xml.sax.ContentHandler):
		def __init__(self, rawDocument):
			xml.sax.ContentHandler.__init__(self)
			self.path = []
			self.document = rawDocument

		def startElement(self, name, attrs):
			self.path.append(name)
			
			if len(self.path) <= 1:
				return

			if self.path[1] == "metadata":
				if len(self.path) >= 2 and self.path[-2] == "identifiers" and self.path[-1] == "id":
					self.document.identifiers.append({ 'type' : attrs['type'] })
			elif self.path[1] == "content":
				if name == "math":
					self.document.rawContent.append(DocumentParser.RawDocument.FormulaContent(attrs.get('alttext')))
				elif len(self.path) == 3 and name == "div":
					self.document.rawContent.append(DocumentParser.RawDocument.Paragraph(attrs['id']))

		def endElement(self, name):
			del self.path[-1] 

		def characters(self, content):
			if len(self.path) <= 1:
				return

			if self.path[1] == 'included_sources':
				if self.path[-1] == 'source':
					self.document.includedSources.append(content)
			elif self.path[1] == 'metadata':
				if len(self.path) >= 2 and self.path[-2] == "identifiers" and self.path[-1] == "id":
					self.document.identifiers[-1]['id'] = content.strip()
				elif self.path[-1] == 'arxiv_title':
					self.document.title = content.strip()
				elif self.path[-1] == 'arxiv_abstract':
					if self.document.abstract is None:
						self.document.abstract = content.strip()
					else:
						self.document.abstract += " " + content.strip()
				elif self.path[-1] == 'arxiv_publication_date':
					self.document.rawPublicationDate = content.strip()
				elif len(self.path) >= 2 and self.path[-2] == "languages" and self.path[-1] == "language":
					self.document.languages.append(content.strip())
				elif len(self.path) >= 3 and self.path[-3] == "authors" and self.path[-2] == "author" and self.path[-1] == "name":
					self.document.plainAuthors.append(content.strip())
				elif len(self.path) >= 2 and self.path[-2] == "zb_author_identifiers" and self.path[-1] == "author_identifier":
					self.document.authorIdentifiers.append(content.strip())
				elif len(self.path) >= 3 and self.path[-3] == "semantic_metadata" and self.path[-2] == "arxiv" and self.path[-1] == "cat":
					self.document.arxivCats.append(content.strip())
				elif len(self.path) >= 3 and self.path[-3] == "semantic_metadata" and self.path[-2] == "zb_msc" and self.path[-1] == "cat":
					self.document.zbMscCats.append(content.strip())

			elif self.path[1] == 'content':
				if not 'math' in self.path:
					if not content.strip() == '':
						self.document.rawContent.append(DocumentParser.RawDocument.TextContent(content))

	class TextTokenizer:
		def __init__(self):
			self.tokenizer = CountVectorizer(input='content', lowercase=True, stop_words='english', min_df=1).build_tokenizer()
			self.wnl = WordNetLemmatizer()
			pass

		def tokenize(self, text):
			return map(
				lambda x: self.wnl.lemmatize(x.lower()),
				self.tokenizer(text)
			)
	
	class FormulaTokenizer:
		def __init__(self):
			pass

		def tokenize(self, matStr):
			if matStr is None or len(matStr) == 0:
				return []
			else:
				tokens = []
				tokenBuffer = None
				strlen = len(matStr)
				state = 0
				index = 0

				while strlen>index:
					char = matStr[index]
					if state==0:
						if char == '\\':
							state = 1
						elif char in DocumentParser.FormulaTokenizer.letters_and_special_chars:
							tokens.append("$" + char + "$")
						else:
							"ignore"
							"""if char in FormulaTokenizer.consciously_ignored:
								"ignore"
							else:
								print "WARNING: ignored character: " + str(char)"""
					elif state==1:
						if char in ascii_letters:
							tokenBuffer = char
							state=2
						else:
							tokens.append("$" + char + "$")
							state=0
					elif state==2:
						if char in ascii_letters:
							tokenBuffer += char
						else:
							tokens.append("$" + tokenBuffer + "$")
							tokenBuffer = None
							index-=1
							state=0
					else:
						raise ValueError("Undefined state while tokenizing")
					index+=1

				if tokenBuffer != None:
					tokens.append("$" + tokenBuffer + "$")

				return tokens

	FormulaTokenizer.consciously_ignored = '{},. %\n:~;$&?`"?@'
	FormulaTokenizer.valid_special_chars = '+-*/_|[]()!<>=^'
	FormulaTokenizer.letters_and_special_chars = ascii_letters + FormulaTokenizer.valid_special_chars + digits
