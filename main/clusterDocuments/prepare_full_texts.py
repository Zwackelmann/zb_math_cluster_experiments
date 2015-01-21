import sys
from os import path
import json
from scipy.sparse import csr_matrix
from util import get_dirpath, get_filenames_and_filepaths, DocumentParser
from util import connect_to_db
from string import printable
import re

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
                if token not in wordCounts:
                    wordCounts[token] = 0
                wordCounts[token] = wordCounts[token] + 1
        count += 1

    return wordCounts


def generateToken2IndexMap(wordCounts, minOccurrence):
    words = sorted(map(lambda x: x[0],
                   filter(lambda x: x[1] >= minOccurrence, wordCounts.items())))
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

    return csr_matrix((data, (row, col)), shape=(numDocs, numAttributes))


def documents2ArffJsonInstancesCorpus(filepaths, tokens2IndexMap):
    p = DocumentParser()

    f = open("raw_data/fulltext-corpus.json", "w")
    f.write("{" +
            "relation-name\":\"full-text-corpus\"," +
            "num-attributes\":" + str(len(tokens2IndexMap)) +
            "}\n")

    for filepath in filepaths:
        doc = p.parse(filepath)
        if "zbmath metadata" in doc.includedSources:
            f.write(doc.toArffJsonDocument(tokens2IndexMap) + "\n")
            f.flush()
    f.close()


# definde paragraph_id_matchers
def matchTopLevelParagraph(string):
    m = re.match(r"p(?P<paragraph>\d+)", string)
    if m:
        return {"type": "top_level_paragraph",
                "paragraph": m.group("paragraph")}
    else:
        return None


def matchSectionParagraph(string):
    m = re.match(r"(c(?P<chapter>\d+)\.)?(s|ch|a)(x)?(?P<section>\d+)(\.p(?P<paragraph>\d+))?", string)
    if m:
        return {"type": "section_paragraph",
                "section": m.group("section"),
                "paragraph": m.group("paragraph")}
    else:
        return None


def matchSubsectionParagraph(string):
    m = re.match(r"(c(?P<chapter>\d+)\.)?(s|ch|a)(x)?(?P<section>\d+)\.(ss)?(x)?(?P<subsection>\d+)(\.p(?P<paragraph>\d+))?", string)
    if m:
        return {
            "type": "subsection_paragraph",
            "section": m.group("section"),
            "subsection": m.group("subsection"),
            "paragraph": m.group("paragraph")}
    else:
        return None


def matchSubsubsectionParagraph(string):
    m = re.match(r"(c(?P<chapter>\d+)\.)?(s|ch|a)(x)?(?P<section>\d+)\.(ss)?(x)?(?P<subsection>\d+)\.(sss)?(x)?(?P<subsubsection>\d+)(\.(p)?(?P<paragraph>\d+))?", string)
    if m:
        return {
            "type": "subsubsection_paragraph",
            "section": m.group("section"),
            "subsection": m.group("subsection"),
            "subsubsection": m.group("subsubsection"),
            "paragraph": m.group("paragraph")}
    else:
        return None


def matchSubsubsectionParagraphParagraph(string):
    m = re.match(r"(c(?P<chapter>\d+)\.)?(s|ch|a)(x)?(?P<section>\d+)\.(ss)?(x)?(?P<subsection>\d+)\.(sss)?(x)?(?P<subsubsection>\d+)\.(p)?(x)?(?P<latex_paragraph>\d+)\.(p)(?P<paragraph>\d+)", string)
    if m:
        return {
            "type": "subsubsection_paragraph_paragraph",
            "section": m.group("section"),
            "subsection": m.group("subsection"),
            "subsubsection": m.group("subsubsection"),
            "paragraph": m.group("paragraph")}
    else:
        return None


def matchAppendixParagraph(string):
    m = re.match(r"a(x)?(?P<appendix>\d+)\.p(?P<paragraph>\d+)", string)
    if m:
        return {
            "type": "appendix_paragraph",
            "appendix": m.group("appendix"),
            "paragraph": m.group("paragraph")}
    else:
        return None


def matchItemizeParagraph(string):
    m = re.match(r"((ch|a|s)\d+\.)?i(?P<itemize>\d+)\.i(x)?(?P<item>\d+)\.p(x)?(?P<paragraph>\d+)", string)
    if m:
        return {
            "type": "itemize",
            "itemize": m.group("itemize"),
            "item": m.group("item"),
            "paragraph": m.group("paragraph")}
    else:
        return None


def matchSubItemizeParagraph(string):
    m = re.match(r"i(?P<itemize>\d+)\.i(?P<subitemize>\d+)\.i(x)?(?P<item>\d+)\.p(x)?(?P<paragraph>\d+)", string)
    if m:
        return {
            "type": "itemize",
            "itemize": str(
                m.group("itemize") +
                "." +
                m.group("subitemize")),
            "item": m.group("item"),
            "paragraph": m.group("paragraph")}
    else:
        return None


def matchIdParagraphRe(string):
    m = re.match(r"id(?P<number>\d+)", string)
    if m:
        return {"type": "id", "number": m.group("number")}
    else:
        return None


def matchTheoremParagraph(string):
    m = re.match(r"(c(?P<chapter>\d+)\.)?((?P<section_type>s|a|ch|c)(x)?(?P<section>\d+)\.)?thm(?P<type>prop|pp|lemma|lem|le|lm|sublemma|theorem|thm|teo|thmspec|letterthm|theo|theof|th|maintheorem|te|mainth|thma|firstthm|introthm|corollary|cor|coro|co|maincoro|example|expl|exm|proposition|pr|prop|propo|definition|df|dfn|def|defin|defn|remark|uremark|rem|rmk|re|remk|question|problem|notation|claim|clm|acknowledgment|acknowledgement|acknow|ack|ass|asn|assumption|asm|prox|conx|res|fact|conj|app|note|dref|hyp|stat|classics|conjecture|pf|algorithm|alg|supplement|ac|proof|issue|oz|demo)?(x|f|s|e)?([a-z]x)?(?P<theorem_nr>\d+(\.\d+)?)(\.p(?P<paragraph>\d+))?", string)
    if m:
        mappings = {
            "prop": "proposition",
            "pr": "proposition",
            "pp": "proposition",
            "propo": "proposition",
            "thm": "theorem",
            "teo": "theorem",
            "thmspec": "theorem",
            "theo": "theorem",
            "theof": "theorem",
            "letterthm": "theorem",
            "te": "theorem",
            "mainth": "theorem",
            "maintheorem": "theorem",
            "thma": "theorem",
            "introthm": "theorem",
            "firstthm": "theorem",
            "dfn": "definition",
            "def": "definition",
            "defin": "definition",
            "defn": "definition",
            "uremark": "remark",
            "rem": "remark",
            "rmk": "remark",
            "re": "remark",
            "remk": "remark",
            "cor": "corollary",
            "coro": "corollary",
            "co": "corollary",
            "maincoro": "corollary",
            "expl": "example",
            "exm": "example",
            "lem": "lemma",
            "le": "lemma",
            "lm": "lemma",
            "sublemma": "lemma",
            "ack": "acknowledgment",
            "acknowledgement": "acknowledgment",
            "acknow": "acknowledgment",
            "ass": "assuption",
            "asm": "assuption",
            "clm": "claim",
            "alg": "algorithm"
        }
        t = m.group("type")
        if t in mappings:
            t = mappings[t]
        return {"type": "theorem", "type_appendix": t}
    else:
        return None


def parseParagraphId(paragraphId):
    matcherList = [matchTopLevelParagraph, matchSectionParagraph, matchSubsectionParagraph, matchSubsubsectionParagraph,
                   matchSubsubsectionParagraphParagraph, matchAppendixParagraph, matchItemizeParagraph,
                   matchSubItemizeParagraph, matchIdParagraphRe, matchTheoremParagraph]

    match = None
    for matcher in matcherList:
        match = matcher(paragraphId)
        if match is not None:
            break

    return match

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

paragraph2InsertStmt = """
INSERT INTO paragraph2(document, paragraph_id, text)
VALUES
(%(document_id)s, %(paragraph_id)s, %(text)s)
"""

theoremInsertStmt = """
INSERT INTO theorem(document, paragraph_id, theorem_type, text)
VALUES
(%(document_id)s, %(paragraph_id)s, %(theorem_type)s, %(text)s)
"""

authorshipInsertStmt = """
INSERT INTO authorship(document, rank, display_name, zbmath_id)
VALUES
(%(document_id)s, %(rank)s, %(display_name)s, %(zbmath_id)s)
"""

mscAssignmentInsertStmt = """
INSERT INTO msc_assignment(document, msc, pos)
VALUES
(%(document_id)s, %(msc)s, %(pos)s)
"""

db = connect_to_db()
cursor = db.cursor()
warning_log = open("warning_log", "a")

p = DocumentParser()
# filepath = "raw_data/test_documents/07040005.xml"
# for filename in filesInDict("raw_data/test_documents", True):
for filename, filepath in zip(filenames, filepaths):
    sys.stdout.write("processing " + filename + "... ")

    # doc, tokenizedParagraphs, formulaDict = p.parseWithParagraphStructure(filename)
    doc, raw_paragraphs, formula_dict = p.parse_raw(filepath)

    # info for doc table:
    document_id = doc.arxiv_id()
    publication_date = doc.publication_date
    title = doc.title
    msc_cats = doc.zb_msc_cats
    main_msc_cat = None if len(doc.zb_msc_cats) == 0 else doc.zb_msc_cats[0][:2]
    authors = doc.authors

    """documentContentMap = {
        "id" : document_id,
        "title" : title,
        "main_msc_cat" : mainMscCat,
        "publication_date" : time.strftime("%Y-%m-%d", publicationDate)
    }

    cursor.execute(documentInsertStmt, documentContentMap)"""

    """formula_id_set = set()
    # formulas
    for formula_id, formula in formulaDict.items():
        formulaContentMap = {
            "document_id" : document_id,
            "formula_id" : formula_id,
            "latex" : formula.latex,
            "p_math_ml" : formula.pMathML,
            "c_math_ml" : formula.cMathML
        }
        cursor.execute(formulaInsertStmt, formulaContentMap)"""

    # paragraphs
    """for paragraph_id, paragraph_array in tokenizedParagraphs:
        paragraphContentMap = {
            "document_id" : document_id,
            "paragraph_id" : paragraph_id,
            "numpy_array" : numpyArr2Bin(paragraph_array)
        }

        cursor.execute(paragraphInsertStmt, paragraphContentMap)

    """
    # paragraph2s
    for paragraph_id, text in raw_paragraphs:
        paragraphContentMap = {
            "document_id": filter(lambda c: c in printable, document_id),
            "paragraph_id": filter(lambda c: c in printable, paragraph_id),
            "text": unicode(text).encode('utf16')
        }

        cursor.execute(paragraph2InsertStmt, paragraphContentMap)

    # authorships
    """for author, rank in zip(authors, range(len(authors))):
        authorId = author.ident
        display_name = author.name

        authorshipContentMap = {
            "document_id": document_id,
            "rank": rank+1,
            "display_name": display_name,
            "zbmath_id": authorId
        }
        cursor.execute(authorshipInsertStmt, authorshipContentMap)"""

    # msc cats
    """for msc_cat, pos in zip(list(set(msc_cats)), range(len(msc_cats))):
        mscAssignmentContentMap = {
            "document_id": document_id,
            "msc": msc_cat,
            "pos": pos+1
        }
        cursor.execute(mscAssignmentInsertStmt, mscAssignmentContentMap)"""

    # theorems
    """for paragraph_id, text in raw_paragraphs:
        x = parseParagraphId(paragraph_id.lower())
        if x is not None and x['type'] == 'theorem':
            theoremContentMap = {
                "document_id": filter(lambda c: c in printable, document_id),
                "paragraph_id": filter(lambda c: c in printable, paragraph_id),
                "theorem_type": x['type_appendix'],
                "text": unicode(text).encode('utf16')
            }

            cursor.execute(theoremInsertStmt, theoremContentMap)"""

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
