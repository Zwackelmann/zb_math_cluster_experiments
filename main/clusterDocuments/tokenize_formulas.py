from util import connectToDb, getAllDocumentIds, filesInDict, build_csr_matrix, save_csr_matrix, load_csr_matrix, addToDict, numpyArr2Bin, bin2NumpyArr
import xml.sax
from xml.sax import saxutils
import json
from string import digits, ascii_letters
import re
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

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

    # calc formula token counts
    """files = filesInDict("derived_data/formula_features", True)
    formulaFeatureMaps = map(lambda file : json.load(open(file)), files)

    tokenCounts = { }
    for fMap in formulaFeatureMaps:
        addToDict(tokenCounts, aggregateFormulaFeatureMaps(
            fMap = fMap,
            onlyTheorems = True,
            ignoreNumbers = False
        ))

    f = open("derived_data/theorem_formula_token_counts.json", "w")
    f.write(json.dumps(tokenCounts))
    f.close()"""

    # build formula token dict
    """minTokenCount = 5

    tokenCounts = json.load(open("derived_data/theorem_formula_token_counts.json"))
    frequentTokens = map(lambda i: i[0], filter(lambda c : c[1] >= minTokenCount, tokenCounts.items()))
    token2IndexMap = dict(zip(sorted(frequentTokens), range(len(frequentTokens))))

    f = open("derived_data/theorem_formula_token2index_map.json", "w")
    f.write(json.dumps(token2IndexMap))
    f.close()"""

    # create raw csr_matrix for theorem formulas
    """files = filesInDict("derived_data/formula_features", True)
    token2IndexMap = json.load(open("derived_data/theorem_formula_token2index_map.json"))
    formulaFeatureMaps = map(lambda file : json.load(open(file)), files)
    
    l = []
    for fMap in formulaFeatureMaps:
        l.append(aggregateFormulaFeatureMaps(
            fMap = fMap,
            onlyTheorems = True,
            ignoreNumbers = False
        ))

    m = build_csr_matrix(listOfMaps=l, token2IndexMap=token2IndexMap)
    save_csr_matrix(m, "derived_data/raw_formula_tdf")"""

    # train and dump tf-idf model for formulas
    """raw_formula_tdf = load_csr_matrix("derived_data/raw_formula_tdf.npz")
    tfidf_trans = TfidfTransformer()
    tfidf_trans.fit(raw_formula_tdf)

    joblib.dump(tfidf_trans, "models/raw_formula_tfidf_model")"""
