from util import connectToDb, getAllDocumentIds, build_csr_matrix, save_csr_matrix, load_csr_matrix, addToDict, numpyArr2Bin, bin2NumpyArr
import json
from string import digits, ascii_letters
import re
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
import numpy as np

db = connectToDb()
cursor = db.cursor()

updateStmt = """
    UPDATE paragraph SET numpy_array = %(numpy_array)s
    WHERE document = %(document_id)s AND paragraph_id = %(paragraph_id)s
"""

selectStmt = """
    SELECT document, paragraph_id, numpy_array 
    FROM paragraph
    WHERE document = %(document_id)s
"""

docIds = getAllDocumentIds(cursor)
count = 0
for docId in docIds:
    cursor.execute(selectStmt, {"document_id" : docId})
    for documentId, paragraphId, numpyArray in cursor:
        par = bin2NumpyArr(numpyArray)

        newPar = []
        for sent in par:
            newSent = []
            for term in sent:
                newTerm = None
                if term[:9] == "<fid fid ":
                    newTerm = "<fid " + term[9:-1] + ">"
                else:
                    newTerm = term

                newSent.append(newTerm)
            newPar.append(newSent)

        newNpArray = numpyArr2Bin(np.array(newPar))

        cursor.execute(updateStmt, {
            "document_id" : documentId, 
            "paragraph_id" : paragraphId,
            "numpy_array" : newNpArray
        })
    count += 1
    print str(count) + " documents processed"
    db.commit()

db.close()