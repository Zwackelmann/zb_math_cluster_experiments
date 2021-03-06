import json
from scipy.sparse import csr_matrix
from util import save_csr_matrix, load_csr_matrix, get_dirpath, get_filenames_and_filepaths, DocumentParser, filesInDict
from util import connectToDb, bin2NumpyArr
import numpy as np
from time import time
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from string import digits, ascii_letters
from os.path import isfile, join

dirpath = get_dirpath()
filenames, filepaths = get_filenames_and_filepaths("raw_data/ntcir_filenames")

tdm = load_csr_matrix("derived_data/zb_math_full_text_tdm2.npz")
translateMap = json.load(open("derived_data/zb_math_full_texts_tokens2IndexMap"))
row_number2fulltext_id_map = json.load(open("derived_data/row_number2fulltext_id_map.json"))

phrase = "theorem"
tokenizer = DocumentParser.TextTokenizer()
tokens = tokenizer.tokenize(phrase)
tokenIds = map(lambda token: translateMap[token], tokens)

candidateIds = []
index = 0
m = tdm[:,tokenIds]

candidateInd = [ ]
currInd = 0
for i in range(len(m.indptr)-1):
    diff = m.indptr[i+1] - m.indptr[i]
    if diff == len(tokenIds):
        candidateInd.append(currInd)
    currInd += 1

print str(len(candidateInd)) + " Candidates\n"

candidateIds = map(lambda i : row_number2fulltext_id_map[str(i)], candidateInd)
candidateDocumentFilenames = map(lambda id : filter(lambda c : c in ascii_letters or c in digits, id) + ".xml.npy", candidateIds) 
candidateDocumentFilepaths = map(lambda filename : join("derived_data/full_text_arrays", filename), candidateDocumentFilenames)


def sentHasTokenSequence(sent, tokens):
    matches = 0
    for token in sent:
        if token == tokens[matches]:
            matches += 1
        else:
            matches = 0
        
        if matches == len(tokens):
            return True

    return False

def parHasTokenSequence(par, tokens):
    for sent in par:
        if sentHasTokenSequence(sent, tokens):
            return True
    return False

def formulasInSentence(sent):
    return filter(lambda token : len(token)>=2 and token[0] == "$" and token[-1] == "$", sent)

def formulasInPar(par):
    return [formula for fsent in map(lambda sent : formulasInSentence(sent), par) for formula in fsent]

def formulasInDoc(doc):
    return [formula for fpar in map(lambda par : formulasInPar(par), doc) for formula in fpar]

"""candidateDocumentFilepaths = filesInDict("derived_data/full_text_arrays", True)[:30]
formulasInDocs = map(lambda path : formulasInDoc(np.load(path)), candidateDocumentFilepaths)

print "\n".join([formula for doc in formulasInDocs for formula in doc])"""

db = connectToDb()
cursor = db.cursor()

getParsStmt = """
    SELECT paragraph_id, numpy_array FROM paragraph
    WHERE document = %(document_id)s
"""

def dumpPar(par):
    return ". ".join(map(lambda sent: " ".join(sent), par))

f = file("related_to_the_work_log", "w")
for docId in candidateIds:
    cursor.execute(getParsStmt, { "document_id" : docId })

    pars = []
    for row in cursor:
        ident = row[0]
        par = bin2NumpyArr(row[1])
        if parHasTokenSequence(par, tokens):
            f.write(str(docId) + ", " + str(ident) + ", " + dumpPar(par) + "\n\n")
            print str(docId) + ", " + str(ident) + ", " + dumpPar(par) + "\n"

            f.flush()

f.close()





