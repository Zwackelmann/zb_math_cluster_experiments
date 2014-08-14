import xml.sax
from string import ascii_letters
from string import digits
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class Document:
	def __init__(self, includedSources, tokens):
		self.includedSources = includedSources
		self.tokens = tokens

	def toDataMap(self, token2IndexMap):
		dataMap = dict()
		for t in self.tokens:
			if t in token2IndexMap:
				i = token2IndexMap[t]
				if not i in dataMap:
					dataMap[i] = 0

				dataMap[i] = dataMap[i] + 1

		return dataMap

	

class DocumentParser:
	def __init__(self):
		self.textTokenizer = DocumentParser.TextTokenizer()
		self.formulaTokenizer = DocumentParser.FormulaTokenizer()

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
			else:
				raise ValueError(str(type(content)) + " is not supported")

		return Document(
			includedSources = rawDocument.includedSources,
			tokens = tokens
		)

	class RawDocument:
		def __init__(self):
			self.includedSources = []
			self.rawContent = []

		class FormulaContent(object):
			def __init__(self, content):
				self.content = content

		class TextContent(object):
			def __init__(self, content):
				self.content = content

	class ZbMathContentHandler(xml.sax.ContentHandler):
		def __init__(self, rawDocument):
			xml.sax.ContentHandler.__init__(self)
			self.path = []
			self.document = rawDocument

		def startElement(self, name, attrs):
			self.path.append(name)
			if name=="math":
				self.document.rawContent.append(DocumentParser.RawDocument.FormulaContent(attrs.get('alttext')))

			# if 'content' in self.path and not 'math' in self.path and not 'table' in self.path:
			# 	"ignore"

		def endElement(self, name):
			del self.path[-1]
			# if 'content' in self.path and not 'math' in self.path and not name == 'math' and not 'table' in self.path and not name == 'table':
			# 	"ignore"

		def characters(self, content):
			if self.path[-1] == 'source':
				self.document.includedSources.append(content)

			if 'content' in self.path and not 'math' in self.path:
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