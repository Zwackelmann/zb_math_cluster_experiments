from util import connectToDb
import xml.sax
from xml.sax import saxutils
import json

class CMathMlTokenizer(xml.sax.ContentHandler):
	def __init__(self, featureMap):
		xml.sax.ContentHandler.__init__(self)
		self.path = []
		self.readApply = False
		self.featureMap = featureMap
		self.ciBuffer = []
		self.cnBuffer = []

	def startElement(self, name, attrs):
		self.path.append(name)
		if name=='apply':
			self.readApply = True
		elif self.readApply:
			key = "opr:" + name
			if not key in self.featureMap:
				self.featureMap[key] = 0
			self.featureMap[key] = self.featureMap[key] + 1

			self.readApply = False

	def endElement(self, name):
		del self.path[-1]

		if name == "ci":
			key = "sym:" + "".join(self.ciBuffer)
			if not key in self.featureMap:
				self.featureMap[key] = 0
			self.featureMap[key] = self.featureMap[key] + 1
			self.ciBuffer = []

		if name == "cn":
			key = "num:" + "".join(self.cnBuffer)
			if not key in self.featureMap:
				self.featureMap[key] = 0
			self.featureMap[key] = self.featureMap[key] + 1
			self.cnBuffer = []

	def characters(self, content):
		if self.path[-1] == "ci":
			self.ciBuffer.append(content)
		elif self.path[-1] == "cn":
			self.cnBuffer.append(content)
		
db = connectToDb()
cursor = db.cursor()

cursor.execute("""
	SELECT id from document
""")

documentIds = []
for row in cursor:
	documentIds.append(row[0])

for documentId in documentIds:
	print documentId

	cursor.execute("""
		SELECT formula_id, c_math_ml 
		FROM formula
		WHERE document = %(document_id)s
	""", {"document_id" : documentId}
	)

	formulaFeatureMap = { }
	for row in cursor:
		formulaId = row[0]
		formulaCMathMl = row[1]

		featureMap = { }
		ch = CMathMlTokenizer(featureMap)
		xml.sax.parseString(formulaCMathMl, ch)

		formulaFeatureMap[formulaId] = featureMap

	f = open("derived_data/formula_features/" + documentId + ".json", "w")
	f.write(json.dumps(formulaFeatureMap))
	f.close()

db.close()
