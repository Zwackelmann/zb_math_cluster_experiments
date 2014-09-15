from util import connectToDb, getAllDocumentIds, filesInDict
import xml.sax
from xml.sax import saxutils
import json
from string import digits, ascii_letters
import re

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

def formula2FeatureMap(formulaCMathMl):
    featureMap = { }
    ch = CMathMlTokenizer(featureMap)
    xml.sax.parseString(formulaCMathMl, ch)
    return featureMap

def prepareFormulaFeaturesForDocument(dbCursor, documentId):
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
        formulaFeatureMap[formulaId] = formula2FeatureMap(formulaCMathMl)

    return formulaFeatureMap

def addToDict(dict1, dict2):
    for k, v in dict2.items():
        if not k in dict1:
            dict1[k] = 0
        dict1[k] += v

def aggregateFormulaFeatureMaps(fMap, onlyTheorems, ignoreNumbers):
    documentFeatureMap = { }
    for fid, features in fMap.items():
        if ignoreNumbers:
            features = dict(filter(lambda kv : not(kv[0][:3] == "num"), features.items()))
        if not onlyTheorems or re.search(r"thm", fid.lower()):
            addToDict(documentFeatureMap, features)

    return documentFeatureMap

if __name__ == "__main__":
    # dump all formula feature maps
    """db = connectToDb()
    cursor = db.cursor()
    documentIds = getAllDocumentIds(cursor)

    for documentId in documentIds:
        print documentId
        formulaFeatureMap = prepareFormulaFeaturesForDocument(cursor, documentId)

        f = open("derived_data/formula_features/" + filter(lambda c : c in ascii_letters or c in digits, documentId) + ".json", "w")
        f.write(json.dumps(formulaFeatureMap))
        f.close()

    db.close()"""

    # build formula token dict
    files = filesInDict("derived_data/formula_features", True)
    formulaFeatureMaps = map(lambda file : json.load(open(file)), files)

    distinctTokens = set()
    for fMap in formulaFeatureMaps:
        distinctTokens.update(aggregateFormulaFeatureMaps(
            fMap = fMap,
            onlyTheorems = True,
            ignoreNumbers = False
        ).keys())

    token2IndexMap = dict(zip(sorted(distinctTokens), range(len(distinctTokens))))
    f = open("derived_data/theorem_formula_terms2index_map.json", "w")
    f.write(json.dumps(token2IndexMap))
    f.close()
    

    # create raw csr_matrix for theorem formulas
