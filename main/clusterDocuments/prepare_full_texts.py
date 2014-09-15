import sys
from os import path
from os import listdir
import json
from scipy.sparse import csr_matrix
from util import save_csr_matrix, load_csr_matrix, get_dirpath, get_filenames_and_filepaths, Document, DocumentParser, filesInDict, connectToDb
import time
import uuid
import numpy as np
import time
import io
from string import printable

dirpath = get_dirpath()
filenames, filepaths = get_filenames_and_filepaths("raw_data/ntcir_filenames")

def buildWordCountDict(filepaths):
    p = DocumentParser()

    wordCounts = dict()
    count = 0
    total = len(filepaths)
    for filepath in filepaths:
        print str(count) + "/" + str(total)
        doc = p.parse(filepath)

        if "zbmath metadata" in doc.includedSources:
            for token in doc.tokens:
                if not token in wordCounts:
                    wordCounts[token] = 0
                wordCounts[token] = wordCounts[token] + 1
        count += 1

    return wordCounts

def generateToken2IndexMap(wordCounts, minOccurrence):
    words = sorted(map(lambda x: x[0], filter(lambda x: x[1]>=minOccurrence, wordCounts.items())))
    return dict(zip(words, range(len(words))))

def dumpDocumentDataMaps(tokens2IndexMap, filenameFilepathsPairs, targetDir):
    p = DocumentParser()

    count = 0
    totalDocs = len(filenameFilepathsPairs)
    for filename, filepath in filenameFilepathsPairs:
        doc = p.parse(filepath)

        print str(count) + " / " + str(totalDocs)

        if "zbmath metadata" in doc.includedSources:
            dataMap = doc.toDataMap(tokens2IndexMap)

            f = open(path.join(targetDir, filename + ".json"), "w")
            f.write(json.dumps(dataMap))
            f.close()

        count += 1

def documentDataMaps2CsrMatrix(filepaths, numAttributes):
    row = []
    col = []
    data = []
    numDocs = 0

    for filepath in filepaths:
        numDocs += 1
        dataMap = json.load(open(filepath))

        for key, val in sorted(dataMap.items(), key=lambda x: int(x[0])):
            row.append(numDocs-1)
            col.append(key)
            data.append(val)

    return csr_matrix( (data,(row,col)), shape=(numDocs, numAttributes) )

def documents2ArffJsonInstancesCorpus(filepaths, tokens2IndexMap):
    p = DocumentParser()

    f = open("raw_data/fulltext-corpus.json", "w")
    f.write("{\"relation-name\":\"full-text-corpus\",\"num-attributes\":" + str(len(tokens2IndexMap)) + "}\n")

    for filepath in filepaths:
        doc = p.parse(filepath)
        if "zbmath metadata" in doc.includedSources:
            f.write(doc.toArffJsonDocument(tokens2IndexMap) + "\n")
            f.flush()
    f.close()

def numpyArr2Bin(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return buffer(out.read())

def bin2NumpyArr(bin):
     return np.load(io.BytesIO(bin))

documentInsertStmt = """
INSERT INTO document(id, title, main_msc_cat, publication_date)
VALUES
(%(id)s, %(title)s, %(main_msc_cat)s, %(publication_date)s)
"""

formulaInsertStmt = """
INSERT INTO formula(document, formula_id, latex, p_math_ml, c_math_ml)
VALUES
(%(document_id)s, %(formula_id)s, %(latex)s, %(p_math_ml)s, %(c_math_ml)s)
"""

paragraphInsertStmt = """
INSERT INTO paragraph(document, paragraph_id, numpy_array)
VALUES
(%(document_id)s, %(paragraph_id)s, %(numpy_array)s)
"""

db = connectToDb()
cursor = db.cursor()
warning_log = open("warning_log", "a")

p = DocumentParser()
# filepath = "raw_data/test_documents/07040005.xml"
# for filename in filesInDict("raw_data/test_documents", True):
for filename, filepath in zip(filenames, filepaths):
    sys.stdout.write("processing " + filename + "... ")

    doc, tokenizedParagraphs, formulaDict = p.parseWithParagraphStructure(filepath)

    # info for doc table:
    document_id = doc.arxivId()
    publicationDate = doc.publicationDate
    title = doc.title
    mainMscCat = None if len(doc.zbMscCats) == 0 else doc.zbMscCats[0][:2]

    documentContentMap = {
        "id" : document_id,
        "title" : title,
        "main_msc_cat" : mainMscCat,
        "publication_date" : time.strftime("%Y-%m-%d", publicationDate)
    }

    cursor.execute(documentInsertStmt, documentContentMap)

    formula_id_set = set()
    # formulas
    for formula_id, formula in formulaDict.items():
        formulaContentMap = {
            "document_id" : document_id,
            "formula_id" : formula_id,
            "latex" : formula.latex,
            "p_math_ml" : formula.pMathML,
            "c_math_ml" : formula.cMathML
        }
        cursor.execute(formulaInsertStmt, formulaContentMap)

    #paragraphs
    for paragraph_id, paragraph_array in tokenizedParagraphs:
        paragraphContentMap = {
            "document_id" : document_id,
            "paragraph_id" : paragraph_id,
            "numpy_array" : numpyArr2Bin(paragraph_array)
        }
        
        cursor.execute(paragraphInsertStmt, paragraphContentMap)
        
    db.commit()
    sys.stdout.write("SUCCESS\n")    

db.close()

"""
# save word count dict
wordCounts = buildWordCountDict(filepaths)
f = open("wordCountsFullTexts", "w")
f.write(json.dumps(wordCounts))
f.close()
"""

"""
# save token-to-index map
wordCounts = json.load(open("wordCountsFullTexts"))
tokens2IndexMap = generateToken2IndexMap(wordCounts, 5)
f = open("zb_math_full_texts_tokens2IndexMap", "w")
f.write(json.dumps(tokens2IndexMap))
f.close()
"""

"""
# dump intermediate full text data maps
tokens2IndexMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
dumpDocumentDataMaps(tokens2IndexMap, zip(filenames, filepaths), "full_text_term_value_maps")
"""

"""
# dump arff json corpus
tokens2IndexMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
documents2ArffJsonInstancesCorpus(filepaths, tokens2IndexMap)
"""

# transform intermediate full text data maps into a csr_matrix
"""tokens2IndexMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
filepaths = [ path.join("derived_data/full_text_term_value_maps", f) for f in listdir("derived_data/full_text_term_value_maps") if path.isfile(path.join("derived_data/full_text_term_value_maps", f)) ]
matrix = documentDataMaps2CsrMatrix(filepaths, len(tokens2IndexMap))
save_csr_matrix(matrix, "derived_data/zb_math_full_text_tdm")"""

"""
# load term-document-matrix
matrix = load_csr_matrix("zb_math_full_text_tdm.npz")
"""
