from util import connectToDb, getAllDocumentIds, build_csr_matrix, save_csr_matrix, load_csr_matrix, addToDict, numpyArr2Bin, bin2NumpyArr, flatten
import json
from string import digits, ascii_letters
import re
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

def getParsFromDocument(documentId, cursor):
    cursor.execute("""
        SELECT paragraph_id, numpy_array FROM paragraph
        WHERE document = %(document)s
    """, {"document" : documentId})

    pars = { }
    for row in cursor:
        pars[row[0]] = bin2NumpyArr(row[1])

    return pars

def parsToFeatureCounts(pars, onlyTheorems):
    thmPars = None
    if onlyTheorems:
        thmPars = map(lambda x: x[1], filter(lambda par: re.search(r"thm", par[0]), pars.items()))
    else:
        thmPars = map(lambda x: x[1], pars.items())

    textTokenList = filter(lambda token : not(token[:5] == "<fid "), flatten(flatten(thmPars)))

    tokenCounts = { }
    for token in textTokenList:
        if not token in tokenCounts:
            tokenCounts[token] = 0
        tokenCounts[token] = tokenCounts[token] + 1

    return tokenCounts

if __name__ == "__main__":
    # calc word counts
    """db = connectToDb()
    cursor = db.cursor()
    documentIds = getAllDocumentIds(cursor)

    globalTokenCounts = { }
    for documentId in documentIds[:10]:
        pars = getParsFromDocument(documentId, cursor)

        featureMap = parsToFeatureCounts(
            pars = pars,
            onlyTheorems = True
        )

        thmPars = map(lambda x: x[1], filter(lambda par: re.search(r"thm", par[0]), pars.items()))
        textTokenList = filter(lambda token : not(token[:5] == "<fid "), flatten(flatten(thmPars)))

        addToDict(globalTokenCounts, featureMap)
    
    f = open("derived_data/theorem_text_token_counts.json", "w")
    f.write(json.dumps(globalTokenCounts))
    f.close()

    db.close()"""

    # build text token dict
    """minTokenCount = 5

    tokenCounts = json.load(open("derived_data/theorem_text_token_counts.json"))
    frequentTokens = map(lambda i: i[0], filter(lambda c : c[1] >= minTokenCount, tokenCounts.items()))
    token2IndexMap = dict(zip(sorted(frequentTokens), range(len(frequentTokens))))

    f = open("derived_data/theorem_text_token2index_map.json", "w")
    f.write(json.dumps(token2IndexMap))
    f.close()"""

    # create raw csr_matrix for theorem texts
    """db = connectToDb()
    cursor = db.cursor()
    documentIds = getAllDocumentIds(cursor)

    token2IndexMap = json.load(open("derived_data/theorem_text_token2index_map.json"))
    
    l = []
    count = 0
    for docId in documentIds[:10]:
        print str(count) + " docs read"
        pars = getParsFromDocument(docId, cursor)

        featureMap = parsToFeatureCounts(
            pars = pars,
            onlyTheorems = True
        )

        l.append(featureMap)
        count += 1

    m = build_csr_matrix(listOfMaps=l, token2IndexMap=token2IndexMap)
    save_csr_matrix(m, "derived_data/raw_theorem_text_tdm")"""

    # train and dump tf-idf model for theorem texts
    """raw_theorem_text_tdm = load_csr_matrix("derived_data/raw_theorem_text_tdm.npz")
    tfidf_trans = TfidfTransformer()
    tfidf_trans.fit(raw_theorem_text_tdm)

    joblib.dump(tfidf_trans, "models/raw_theorem_text_tfidf_model")"""