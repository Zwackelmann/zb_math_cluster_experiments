from util import connectToDb, getAllDocumentIds, build_csr_matrix, save_csr_matrix, load_csr_matrix, addToDict, numpyArr2Bin, bin2NumpyArr, flatten
import json
from string import digits, ascii_letters
import re
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from tokenize_paragraphs import getParsFromDocument, parsToFeatureCounts
from sklearn.preprocessing import Normalizer

def getFormulaIdsFromPars(pars, onlyTheorems):
    thmPars = None
    if onlyTheorems:
        thmPars = map(lambda x: x[1], filter(lambda par: re.search(r"thm", par[0]), pars.items()))
    else:
        thmPars = map(lambda x: x[1], pars.items())

    formulaTokens = filter(lambda token : token[:5] == "<fid ", flatten(flatten(thmPars)))

    return map(lambda token: token[5:-1], formulaTokens)

def formulasToFeatureCounts(formulaIds):
    formulaFile = "derived_data/formula_features/" + filter(lambda c : c in (digits + ascii_letters), documentId) + ".json"
    formulas = json.load(open(formulaFile))

    m = { }
    formulaFeatureList = map(lambda fid : formulas[fid], formulaIds)
    for formulaFeatureMap in formulaFeatureList:
        addToDict(m, formulaFeatureMap)

    return m

def processFeatureCounts(featureCounts, token2Id, tfidfModel):
    m = build_csr_matrix([featureCounts], token2Id)
    return Normalizer().transform(tfidfModel.transform(m))

def combine_csr_matrixes(matrix_list):
    l = matrix_list[0]
    print l.data
    print l.indices
    print l.indptr

db = connectToDb()
cursor = db.cursor()
documentIds = getAllDocumentIds(cursor)

for documentId in [documentIds[3]]:
    pars = getParsFromDocument(documentId, cursor)

    # get text feature map
    textFeatureCounts = parsToFeatureCounts(
        pars = pars,
        onlyTheorems = True
    )

    # get formula feature map
    formulaIds = getFormulaIdsFromPars(
        pars = pars,
        onlyTheorems = True
    )

    formulaFeatureCounts = formulasToFeatureCounts(formulaIds)

    textFeatures = processFeatureCounts(
        featureCounts = textFeatureCounts, 
        token2Id = json.load(open("derived_data/theorem_text_token2index_map.json")),
        tfidfModel = joblib.load("models/raw_theorem_text_tfidf_model")
    )

    formulaFeatures = processFeatureCounts(
        featureCounts = formulaFeatureCounts, 
        token2Id = json.load(open("derived_data/theorem_formula_token2index_map.json")),
        tfidfModel = joblib.load("models/raw_formula_tfidf_model")
    )


    combine_csr_matrixes([textFeatures, formulaFeatures])