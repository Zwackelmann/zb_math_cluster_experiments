import xml.sax
from string import ascii_letters
from string import digits
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import nltk
import re

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
				tokens.append("$" + formulaDict[int(sentence[res.start()+1:res.end()-1])] + "$")
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