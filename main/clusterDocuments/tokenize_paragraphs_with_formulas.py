from util import connectToDb, getAllDocumentIds, build_csr_matrix, save_csr_matrix, load_csr_matrix, addToDict, numpyArr2Bin, bin2NumpyArr, flatten
import json
from string import digits, ascii_letters
import re
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from tokenize_paragraphs import getParsFromDocument, parsToFeatureCounts
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
import numpy as np

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
    m = build_csr_matrix(listOfMaps=[featureCounts], token2IndexMap=token2Id)
    return Normalizer().transform(tfidfModel.transform(m))

def vertically_append_matrix(srcMatrix, appendMatrix):
    numRows = srcMatrix.shape[0]
    numSrcCols = srcMatrix.shape[1]
    totalNumCols = srcMatrix.shape[1]+appendMatrix.shape[1]

    newData = []
    newIndices = []
    newIndptr = [0]

    currSrcOffset = 0
    currAppendOffset = 0
    for currRow in range(numRows):
        nextSrcRow = srcMatrix.indptr[currRow+1]
        nextAppendRow = appendMatrix.indptr[currRow+1]

        newData.extend(srcMatrix.data[currSrcOffset:nextSrcRow])
        newData.extend(appendMatrix.data[currAppendOffset:nextAppendRow])
        newIndices.extend(srcMatrix.indices[currSrcOffset:nextSrcRow])
        newIndices.extend(map(lambda x: x+numSrcCols, appendMatrix.indices[currAppendOffset:nextAppendRow]))
        newIndptr.append(newIndptr[-1]+(nextSrcRow-currSrcOffset)+(nextAppendRow-currAppendOffset))

        currSrcOffset += (nextSrcRow-currSrcOffset)
        currAppendOffset += (nextAppendRow-currAppendOffset)

    return csr_matrix((np.array(newData), np.array(newIndices), np.array(newIndptr)), shape=(numRows, totalNumCols))

def horizontally_combine_matrixes(matrixList):
    if len(matrixList) == 0:
        return build_csr_matrix(listOfMaps = [], numAttributes = 0)
    
    assert len(matrixList) >= 1

    newData = []
    newIndices = []
    newIndptr = [0]

    numNonZerosInMat = map(lambda matrix: len(matrix.data), matrixList)
    numCols = matrixList[0].shape[1]
    numRows = 0

    count = 0
    offset = 0
    for matrix in matrixList:
        if matrix.shape[1] != numCols:
            raise ValueError("Num attributes of matrixes for hirizontal combination must be identical")
        numRows += matrix.shape[0]

        newData.extend(matrix.data)
        newIndices.extend(matrix.indices)
        newIndptr.extend(map(lambda x: x+offset, matrix.indptr[1:]))

        offset += numNonZerosInMat[count]
        count += 1

    return csr_matrix((np.array(newData), np.array(newIndices), np.array(newIndptr)), shape=(numRows, numCols))

if __name__ == "__main__":
    db = connectToDb()
    cursor = db.cursor()
    documentIds = getAllDocumentIds(cursor)

    matrixList = [ ]
    count = 0
    for documentId in documentIds:
        print "read " + str(count) + " documents" 
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

        combinedFeatures = vertically_append_matrix(textFeatures, formulaFeatures)
        matrixList.append(combinedFeatures)

    theoremTDM = horizontally_combine_matrixes(matrixList)
    save_csr_matrix(theoremTDM, "derived_data/combined_theorem_text_formula_tdm")
        