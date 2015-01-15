import numpy as np
from scipy.sparse import csr_matrix
import uuid
from main.arffJson.ArffJsonCorpus import ArffJsonCorpus, ArffJsonDocument
from scipy import stats
import math
from sklearn.feature_selection import chi2
from os import listdir
from os.path import isfile, join
import xml.sax
from string import ascii_letters, digits, printable
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import nltk
import re
from xml.sax import saxutils
from datetime import date
import json
import MySQLdb
import io
import copy
import itertools


# file system
def get_dirpath():
    if uuid.getnode() == 161338626918L:  # is69
        dirpath = "/raid0/barthel/data/NTCIR_2014_enriched/"
    elif uuid.getnode() == 622600420609L or uuid.getnode() == 220964050453213:  # xmg-laptop
        dirpath = "/home/simon/samba/ifis/ifis/Datasets/math_challange/NTCIR_2014_enriched/"
    else:
        raise ValueError("unknown node id " + str(uuid.getnode()))

    return dirpath


def get_filenames_and_filepaths(file):
    dirpath = get_dirpath()

    tmp = [(line.strip(), dirpath + line.strip()) for line in open(file)]
    filenames = map(lambda x: x[0], tmp)
    filepaths = map(lambda x: x[1], tmp)

    return filenames, filepaths


def files_in_dict(path, with_path=False):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]

    if with_path:
        return map(lambda filename: join(path, filename), filenames)
    else:
        return filenames


# csr matrixes
def save_csr_matrix(array, filename):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_csr_matrix(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def build_csr_matrix(list_of_maps, token_2_index_map=None, num_attributes=None):
    if token_2_index_map is None and num_attributes is None:
        raise ValueError("either token_2_index_map or num_attributes must be set")
    elif (token_2_index_map is not None) and (num_attributes is not None):
        raise ValueError("token_2_index_map and num_attributes cannot both be set")
    else:
        pass

    if not type(list_of_maps) is list:
        list_of_maps = [list_of_maps]

    row = []
    col = []
    data = []

    i = 0
    for m in list_of_maps:
        numerical_sorted_tokens = None

        if token_2_index_map is not None:
            tokens_in_dict = filter(lambda kv: kv[0] in token_2_index_map, m.items())
            translated_tokens = map(lambda kv: (token_2_index_map[kv[0]], kv[1]), tokens_in_dict)
            numerical_sorted_tokens = sorted(translated_tokens, key=lambda x: x[0])
        elif num_attributes is not None:
            numerical_sorted_tokens = sorted(m.items(), key=lambda x: x[0])
        else:
            raise ValueError("Invalid case while building csr_matrix")

        for key, val in numerical_sorted_tokens:
            row.append(i)
            col.append(key)
            data.append(val)

        i += 1

    shape_rows = len(list_of_maps)
    shape_cols = None
    if token_2_index_map is not None:
        shape_cols = len(token_2_index_map)
    elif num_attributes is not None:
        shape_cols = num_attributes
    else:
        raise ValueError("Invalid case while building csr_matrix")

    return csr_matrix((data, (row, col)), shape=(shape_rows, shape_cols))


# db
def connect_to_db():
    credentials = json.load(open("db_connect.json"))
    db = MySQLdb.connect(**credentials)
    return db


def get_all_document_ids(cursor):
    cursor.execute("SELECT id from document")

    document_ids = []
    for row in cursor:
        document_ids.append(row[0])

    return document_ids


def numpy_arr_2_bin(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return buffer(out.read())


def bin_2_numpy_arr(bin):
    return np.load(io.BytesIO(bin))


# utility
def flatten(l):
    return [item for sublist in l for item in sublist]


def takeN(iter, n):
    for i in xrange(n):
        yield iter.next()


def print_best_n(dict, n):
    best_n = [(int(k), sorted(v, key=lambda x: x[1], reverse=True)[:n]) for k, v in dict.items()]
    sorted_best_n = sorted(best_n, key=lambda x: int(x[0]))
    return "\n".join(map(lambda x: repr(x), sorted_best_n))


def word_list_rank_correlation(chis1, chis2):
    chi_terms_1 = set(map(lambda x: x[0], chis1))
    chi_terms_2 = set(map(lambda x: x[0], chis2))

    all_terms = chi_terms_1.union(chi_terms_2)
    term_dict = dict(zip(all_terms, range(len(all_terms))))

    chi_list_1 = [0.0] * len(all_terms)
    chi_list_2 = [0.0] * len(all_terms)

    for term, score in chis1:
        chi_list_1[term_dict[term]] = score
    for term, score in chis2:
        chi_list_2[term_dict[term]] = score

    tau, p = stats.spearmanr(chi_list_1, chi_list_2)
    return tau, p


def indexes_in_list(l, items):
    if type(items) is not set:
        items = set(items)

    zipped_with_index = zip(l, range(len(l)))
    filtered = filter(lambda id_ind: id_ind[0] in items, zipped_with_index)
    return map(lambda x: x[1], filtered)


def ascii_escape(str):
    return filter(lambda c: c in printable, str.encode('ascii', 'xmlcharrefreplace'))


def add_to_dict(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = 0
        dict1[k] += v


def group_and_count(l, key=None):
    if key is not None:
        l = map(key, l)
    d = {}
    for item in l:
        if item not in d:
            d[item] = 0
        d[item] = d[item] + 1
    return d


def read_file_linewise_2_array(file):
    f = open(file)
    a = []
    for l in f:
        a.append(l.strip())
    return a


# group documents (by cluster or category)
def get_index_to_word_map(file):
    """TODO: this method seems to be a duplicate of the read_dict(dict_filepath)
    below. Keep only one of those"""
    count = 0
    d = {}
    for line in open(file):
        d[count] = line.strip()
        count += 1
    return d


def read_dict(dict_filepath):
    """TODO: this method seems to be a duplicate of the get_index_to_word_map(file)
    above. Keep only one of those"""
    lines = [line.strip() for line in open(dict_filepath)]
    index = 0
    index_2_word = {}
    for line in lines:
        index_2_word[index] = line
        index += 1
    return index_2_word


def group_cluster_documents(cluster_file):
    cluster_to_documents = {}

    for line in open(cluster_file):
        x = line.split(";")
        doc_id = x[0]
        doc_class = x[1].strip()

        if not (doc_class in cluster_to_documents):
            cluster_to_documents[doc_class] = set()

        cluster_to_documents[doc_class].add(doc_id)

    return cluster_to_documents


def group_corpus_documents(corpus_filepath, mode):
    """
    mode=1: only main classes
    mode=2: also secondary classes
    """
    category_2_documents = {}
    for doc in ArffJsonCorpus(corpus_filepath):
        if len(doc.classes) != 0:
            doc_id = doc.id

            if mode == 1:
                doc_main_class = doc.classes[0][0:2]

                if not (doc_main_class in category_2_documents):
                    category_2_documents[doc_main_class] = set()

                category_2_documents[doc_main_class].add(doc_id)
            elif mode == 2:
                for doc_class in doc.classes:
                    top_class = doc_class[0:2]

                    if not (top_class in category_2_documents):
                        category_2_documents[top_class] = set()

                    category_2_documents[top_class].add(doc_id)

    return category_2_documents


# gen label matrixes (e.g. as preparation for chi square)
def initialize_label_matrix_from_corpus(corpus):
    class_label_2_number = {}
    class_number_2_label = {}
    curr_class_index = 0
    num_docs = 0
    for doc in iter(corpus):
        for cl in doc.classes:
            if not cl[:2] in class_label_2_number:
                class_label_2_number[cl[:2]] = curr_class_index
                class_number_2_label[curr_class_index] = cl[:2]
                curr_class_index += 1

        num_docs += 1

    num_classes = len(class_label_2_number)
    label_matrix = map(lambda x: [0]*num_docs, range(0, num_classes))

    for doc_index, doc in zip(xrange(0, num_docs), iter(corpus)):
        for cl in doc.classes:
            label_matrix[class_label_2_number[cl[:2]]][doc_index] = 1

    return label_matrix, class_label_2_number, class_number_2_label


def initialize_label_matrix_from_clusters(cluster_file):
    num_docs = 0
    clusters = set()

    for line in open(cluster_file):
        x = line.split(";")
        doc_cluster = int(x[1].strip())
        clusters.add(doc_cluster)
        num_docs += 1

    num_clusters = len(clusters)
    label_matrix = map(lambda x: [0]*num_docs, range(0, num_clusters))

    doc_index = 0
    for line in open(cluster_file):
        x = line.split(";")
        doc_cluster = int(x[1].strip())

        label_matrix[int(doc_cluster)][doc_index] = 1
        doc_index += 1

    return label_matrix


# Chi square
def read_chi_file(chi_filepath):
    return map(lambda x: (int(x[0]), float(x[1]), float(x[2])), [line.split(";") for line in open(chi_filepath)])


def dump_chi_scores(TDM, label_matrix, class_number_2_label):
    best_terms = get_chi_scores(TDM, label_matrix, 1000)

    for i in range(0, len(class_number_2_label)):
        cl = class_number_2_label[i]
        best_terms_for_class = best_terms[i]

        file = open("chi-" + str(cl), "w")
        for index, score, p in best_terms_for_class:
            file.write(str(index) + ";" + str(score) + ";" + str(p) + "\n")
        file.close()


def get_chi_scores(TDM, label_matrix, first_n):
    best_terms = []

    for i in range(0, len(label_matrix)):
        v = chi2(TDM, label_matrix[i])
        v = zip(range(0, len(v[0])), v[0], v[1])
        v = filter(lambda x: not(math.isnan(x[1])), v)

        best_terms.append(sorted(v, key=lambda x: x[1], reverse=True)[:first_n])

    return best_terms


def chi_set_geq(folder, threshold):
    chi_set = set()

    for chi_file in folder:
        chis = map(lambda x: x[0], filter(lambda x: x[1] > threshold, read_chi_file(chi_file)))
        chi_set.update(chis)

    return chi_set


def get_chi_files(folder):
    filenames, cat_labels = zip(*[(f, f[4:]) for f in listdir(folder) if isfile(join(folder, f)) and f[:4] == "chi-"])
    return filenames, cat_labels


def get_best_chi_terms(label_matrix, TDM, index_2_word_map, chi_threshold, class_number_2_label=None):
    bestTerms = get_chi_scores(TDM, label_matrix, 10000)

    d = {}
    index = 0
    for best_terms_for_cluster in bestTerms:
        best_terms_for_cluster = filter(lambda x: x[1] > chi_threshold, best_terms_for_cluster)

        if class_number_2_label is None:
            d[index] = map(lambda x: (index_2_word_map[x[0]], x[1]), best_terms_for_cluster)
        else:
            d[class_number_2_label[index]] = map(lambda x: (index_2_word_map[x[0]], x[1]), best_terms_for_cluster)
        index += 1

    return d


# Full text stuff
class Document:
    def __init__(self, identifiers=[], title=None,
                 abstract=None, languages=[],
                 included_sources=[],
                 publication_date=None, zb_msc_cats=[],
                 arxiv_cats=[], authors=[],
                 full_text_tokens=[]):
        self.identifiers = identifiers
        self.title = title
        self.abstract = abstract
        self.languages = languages
        self.included_sources = included_sources
        self.publication_date = publication_date
        self.zb_msc_cats = zb_msc_cats
        self.arxiv_cats = arxiv_cats
        self.authors = authors
        self.full_text_tokens = full_text_tokens

    def arxiv_id(self):
        ids = filter(lambda x: x.source == "arxiv", self.identifiers)

        if len(ids) != 1:
            if len(ids) > 1:
                raise ValueError("Found multiple arxiv ids")
            else:
                raise ValueError("Didn't find any arxiv id")
        else:
            return ids[0].ident

    def to_data_map(self, token_2_index_map):
        data_map = dict()
        for t in self.full_text_tokens:
            if t in token_2_index_map:
                i = token_2_index_map[t]
                if i not in data_map:
                    data_map[i] = 0

                data_map[i] = data_map[i] + 1

        return data_map

    def to_arff_json_document(self, token_2_index_map):
        str_buffer = []
        str_buffer.append("[[\"")

        ids = filter(lambda x: x.source == "arxiv", self.identifiers)

        if len(ids) != 1:
            if len(ids) > 1:
                raise ValueError("Found multiple arxiv ids")
            else:
                raise ValueError("Didn't find any arxiv id")
        else:
            str_buffer.append(ids[0].ident)

        str_buffer.append("\",[")
        str_buffer.append(",".join(map(lambda cl: "\"" + cl + "\"", self.zb_msc_cats)))
        str_buffer.append("]],{")

        data_map = self.to_data_map(token_2_index_map)
        sorted_keys = sorted(data_map.items(), key=lambda x: int(x[0]))
        str_buffer.append(",".join(map(lambda kv: "\"" + str(kv[0]) + "\":" + str(kv[1]), sorted_keys)))

        str_buffer.append("}]")

        return "".join(str_buffer)

    class Identifier:
        def __init__(self, ident, source):
            self.ident = ident
            self.source = source

        def __str__(self):
            return "Identifier(source=" + self.source + ", ident=" + self.ident + ")"

        def __repr__(self):
            return str(self)

    class Author:
        def __init__(self, name, ident=None):
            self.name = name
            self.ident = ident

        def __str__(self):
            return "Author(" + self.name + ("(" + self.ident + ")" if self.ident is not None else "") + ")"

        def __repr__(self):
            return str(self)


class DocumentParser:
    def __init__(self):
        self.textTokenizer = DocumentParser.TextTokenizer()
        self.formulaTokenizer = DocumentParser.FormulaTokenizer()
        self.sentenceDetector = nltk.data.load('tokenizers/punkt/english.pickle')

    def parse(self, filepath):
        source = open(filepath)
        rawDocument = DocumentParser.RawDocument()
        ch = DocumentParser.ZbMathContentHandler(rawDocument)
        xml.sax.parse(source, ch)

        tokens = []
        for content in rawDocument.rawContent:
            if type(content) is DocumentParser.RawDocument.TextContent:
                tokens.extend(self.textTokenizer.tokenize(content.content))
            elif type(content) is DocumentParser.RawDocument.FormulaContent:
                tokens.extend(self.formulaTokenizer.tokenize(content.latex))
            elif type(content) is DocumentParser.RawDocument.Paragraph:
                pass
            else:
                raise ValueError(str(type(content)) + " is not supported")

        rawDocument.rawContent = tokens
        return rawDocument.toDocument()

    def parseWithParagraphStructure(self, filepath):
        source = open(filepath)
        rawDocument = DocumentParser.RawDocument()

        ch = DocumentParser.ZbMathContentHandler(rawDocument)
        xml.sax.parse(source, ch)

        paragraphs = []
        paragraphBuffer = []
        formulaDict = {}
        currentParagraphId = None

        for content in rawDocument.rawContent:
            if type(content) is DocumentParser.RawDocument.TextContent:
                paragraphBuffer.append(content.content)
            elif type(content) is DocumentParser.RawDocument.FormulaContent:
                paragraphBuffer.append("<fid " + content.ident + ">")
                formulaDict[content.ident] = content
            elif type(content) is DocumentParser.RawDocument.Paragraph:
                if currentParagraphId is not None:
                    paragraphString = " ".join(paragraphBuffer)
                    paragraphBuffer = []
                    sentences = self.sentenceDetector.tokenize(paragraphString)

                    paragraphs.append((currentParagraphId, map(lambda s: self.tokenizeSentence(s, None), sentences)))

                currentParagraphId = content.ident
            else:
                raise ValueError(str(type(content)) + " is not supported")

        if len(paragraphBuffer) != 0 and currentParagraphId is not None:
            paragraphString = " ".join(paragraphBuffer)
            sentences = self.sentenceDetector.tokenize(paragraphString)
            paragraphs.append((currentParagraphId, map(lambda s: self.tokenizeSentence(s, None), sentences)))

        return rawDocument.toDocument(), paragraphs, formulaDict

    def parseRaw(self, filepath):
        source = open(filepath)
        rawDocument = DocumentParser.RawDocument()

        ch = DocumentParser.ZbMathContentHandler(rawDocument)
        xml.sax.parse(source, ch)

        paragraphs = []
        paragraphBuffer = []
        formulaDict = {}
        currentParagraphId = None

        for content in rawDocument.rawContent:
            if type(content) is DocumentParser.RawDocument.TextContent:
                paragraphBuffer.append(content.content)
            elif type(content) is DocumentParser.RawDocument.FormulaContent:
                paragraphBuffer.append("<fid " + content.ident + ">")
                formulaDict[content.ident] = content
            elif type(content) is DocumentParser.RawDocument.Paragraph:
                if currentParagraphId is not None:
                    paragraphString = " ".join(paragraphBuffer)
                    paragraphBuffer = []

                    paragraphs.append((currentParagraphId, paragraphString))

                currentParagraphId = content.ident
            else:
                raise ValueError(str(type(content)) + " is not supported")

        if len(paragraphBuffer) != 0 and currentParagraphId is not None:
            paragraphString = " ".join(paragraphBuffer)
            paragraphs.append((currentParagraphId, paragraphString))

        return rawDocument.toDocument(), paragraphs, formulaDict

    def tokenizeSentence(self, sentence, formulaDict=None):
        tokens = []

        while len(sentence) != 0:
            res = re.search(r"<fid [^>]+>", sentence)
            if res is None:
                tokens.extend(self.textTokenizer.tokenize(sentence))
                break
            else:
                tokens.extend(self.textTokenizer.tokenize(sentence[:res.start()]))

                formulaId = sentence[res.start()+5:res.end()-1]
                if formulaDict is not None:
                    formula = formulaDict.get(formulaId)
                    if formula is not None:
                        tokens.append("$" + formula.latex + "$")
                else:
                    tokens.append("<fid " + formulaId + ">")

                sentence = sentence[res.end():]
        return tokens

    class RawDocument:
        def __init__(self):
            self.includedSources = []
            self.identifiers = []
            self.title = None
            self.abstract = None
            self.languages = []
            self.rawPublicationDate = None
            self.zbMscCats = []
            self.arxivCats = []
            self.plainAuthors = []
            self.authorIdentifiers = []
            self.rawContent = []

        class FormulaContent(object):
            def __init__(self, ident, latex, pMathML, cMathML):
                self.ident = ident
                self.latex = latex
                self.pMathML = pMathML
                self.cMathML = cMathML

        class TextContent(object):
            def __init__(self, content):
                self.content = content

        class Paragraph(object):
            def __init__(self, ident):
                self.ident = ident

        def toDocument(self):
            authors = []
            if len(self.authorIdentifiers) == len(self.plainAuthors):
                for name, ident in zip(self.plainAuthors, self.authorIdentifiers):
                    authors.append(Document.Author(name=name, ident=ident))
            else:
                for name in self.plainAuthors:
                    authors.append(Document.Author(name=name, ident=None))

            identifiers = []
            for ident in self.identifiers:
                identifiers.append(Document.Identifier(ident=ident['id'], source=ident['type']))

            parsedTime = None
            if self.rawPublicationDate is not None:
                parsedTime = time.strptime(self.rawPublicationDate[:10], "%Y-%m-%d")

            return Document(
                includedSources=self.includedSources,
                identifiers=identifiers,
                title=self.title,
                abstract=self.abstract,
                languages=self.languages,
                publicationDate=parsedTime,
                zbMscCats=self.zbMscCats,
                arxivCats=self.arxivCats,
                authors=authors,
                fullTextTokens=self.rawContent
            )

    RawDocument.dateFormat = ""

    class ZbMathContentHandler(xml.sax.ContentHandler):
        def __init__(self, rawDocument):
            xml.sax.ContentHandler.__init__(self)
            self.path = []
            self.document = rawDocument

            # mathml capturing
            self.formulaId = None
            self.latexBuffer = None
            self.cMathMLBuffer = []
            self.pMathMLBuffer = []
            self.capturingMathState = None

        def startElement(self, name, attrs):
            self.path.append(name)
            if len(self.path) <= 1:
                return

            if self.path[1] == "metadata":
                if len(self.path) >= 2 and self.path[-2] == "identifiers" and self.path[-1] == "id":
                    self.document.identifiers.append({'type': attrs['type']})
            elif self.path[1] == "content":
                if self.capturingMathState is None:
                    if name == "math":
                        self.formulaId = ascii_escape(attrs['id']) if 'id' in attrs.keys() else None
                        self.latexBuffer = ascii_escape(attrs['alttext']) if 'alttext' in attrs.keys() else None
                        self.capturingMathState = "found math tag"
                    elif len(self.path) == 3 and name == "div":
                        self.document.rawContent.append(DocumentParser.RawDocument.Paragraph(ascii_escape(attrs['id'])))
                else:
                    if self.capturingMathState == "found math tag" and name == "semantics":
                        self.capturingMathState = "capture pmathml"
                    elif self.capturingMathState == "capture pmathml" and name != "annotation-xml":
                        self.pMathMLBuffer.append("<" + name + ">")
                    elif self.capturingMathState == "capture pmathml" and name == "annotation-xml":
                        self.capturingMathState = "capture cmathml"
                    elif self.capturingMathState == "capture cmathml":
                        self.cMathMLBuffer.append("<" + name + ">")
                    elif self.capturingMathState == "fading out":
                        pass
                    else:
                        raise ValueError("WARNING: invalid state while captureing math: " + self.capturingMathState)

        def endElement(self, name):
            del self.path[-1]

            if self.capturingMathState == "capture pmathml":
                self.pMathMLBuffer.append("</" + name + ">")
            elif self.capturingMathState == "capture cmathml" and name != "annotation-xml":
                self.cMathMLBuffer.append("</" + name + ">")
            elif self.capturingMathState == "capture cmathml" and name == "annotation-xml":
                self.capturingMathState = "fading out"
            else:
                pass

            if name == "math":
                if self.formulaId is not None:
                    formula = DocumentParser.RawDocument.FormulaContent(
                        ident=self.formulaId,
                        latex=self.latexBuffer,
                        pMathML="<math>" + "".join(self.pMathMLBuffer) + "</math>",
                        cMathML="<math>" + "".join(self.cMathMLBuffer) + "</math>"
                    )
                    self.document.rawContent.append(formula)

                self.capturingMathState = None
                self.formulaId = None
                self.latexBuffer = None
                self.pMathMLBuffer = []
                self.cMathMLBuffer = []

        def characters(self, content):
            if len(self.path) <= 1:
                return

            if self.path[1] == 'included_sources':
                if self.path[-1] == 'source':
                    self.document.includedSources.append(content)
            elif self.path[1] == 'metadata':
                if len(self.path) >= 2 and self.path[-2] == "identifiers" and self.path[-1] == "id":
                    self.document.identifiers[-1]['id'] = content.strip()
                elif self.path[-1] == 'arxiv_title':
                    if self.document.title is None:
                        self.document.title = content.strip()
                    else:
                        self.document.title += content.strip()
                elif self.path[-1] == 'arxiv_abstract':
                    if self.document.abstract is None:
                        self.document.abstract = content.strip()
                    else:
                        self.document.abstract += " " + content.strip()
                elif self.path[-1] == 'arxiv_publication_date':
                    self.document.rawPublicationDate = content.strip()
                elif len(self.path) >= 2 and self.path[-2] == "languages" and self.path[-1] == "language":
                    self.document.languages.append(content.strip())
                elif len(self.path) >= 3 and self.path[-3] == "authors" and self.path[-2] == "author" and self.path[-1] == "name":
                    self.document.plainAuthors.append(content.strip())
                elif len(self.path) >= 2 and self.path[-2] == "zb_author_identifiers" and self.path[-1] == "author_identifier":
                    self.document.authorIdentifiers.append(content.strip())
                elif len(self.path) >= 3 and self.path[-3] == "semantic_metadata" and self.path[-2] == "arxiv" and self.path[-1] == "cat":
                    self.document.arxivCats.append(content.strip())
                elif len(self.path) >= 3 and self.path[-3] == "semantic_metadata" and self.path[-2] == "zb_msc" and self.path[-1] == "cat":
                    self.document.zbMscCats.append(content.strip())

            elif self.path[1] == 'content':
                if self.capturingMathState is None:
                    if not content.strip() == '':
                        self.document.rawContent.append(DocumentParser.RawDocument.TextContent(content))
                else:
                    if self.capturingMathState == "capture pmathml":
                        self.pMathMLBuffer.append(ascii_escape(saxutils.escape(content.strip())))
                    elif self.capturingMathState == "capture cmathml":
                        self.cMathMLBuffer.append(ascii_escape(saxutils.escape(content.strip())))
                    else:
                        pass

    class TextTokenizer:
        def __init__(self):
            self.tokenizer = CountVectorizer(input='content', lowercase=True, stop_words='english', min_df=1).build_tokenizer()
            self.wnl = WordNetLemmatizer()

        def tokenize(self, text):
            return map(
                lambda x: self.wnl.lemmatize(x.lower()),
                self.tokenizer(text)
            )

    class FormulaTokenizer:
        def __init__(self):
            pass

        def tokenize(self, matStr):
            if matStr is None or len(matStr) == 0:
                return []
            else:
                tokens = []
                tokenBuffer = None
                strlen = len(matStr)
                state = 0
                index = 0

                while strlen > index:
                    char = matStr[index]
                    if state == 0:
                        if char == '\\':
                            state = 1
                        elif char in DocumentParser.FormulaTokenizer.letters_and_special_chars:
                            tokens.append("$" + char + "$")
                        else:
                            pass

                    elif state == 1:
                        if char in ascii_letters:
                            tokenBuffer = char
                            state = 2
                        else:
                            tokens.append("$" + char + "$")
                            state = 0
                    elif state == 2:
                        if char in ascii_letters:
                            tokenBuffer += char
                        else:
                            tokens.append("$" + tokenBuffer + "$")
                            tokenBuffer = None
                            index -= 1
                            state = 0
                    else:
                        raise ValueError("Undefined state while tokenizing")
                    index += 1

                if tokenBuffer is not None:
                    tokens.append("$" + tokenBuffer + "$")

                return tokens

    FormulaTokenizer.consciously_ignored = '{},. %\n:~;$&?`"?@'
    FormulaTokenizer.valid_special_chars = '+-*/_|[]()!<>=^'
    FormulaTokenizer.letters_and_special_chars = ascii_letters + FormulaTokenizer.valid_special_chars + digits


formula_parse_error_count = 0


class FormulaTokenizer:
    nodeCounter = 0

    def tokenize(self, formula):
        global formula_parse_error_count
        ch = FormulaTokenizer.FormulaContentHandler()

        try:
            xml.sax.parseString(formula, ch)
            return FormulaTokenizer.leafsToTokens(ch.leafs)
        except xml.sax._exceptions.SAXParseException:
            formula_parse_error_count += 1
            print "Formula parse error! (" + str(formula_parse_error_count) + ")"
            return []

    @classmethod
    def leafsToTokens(cls, leafs):
        opathLists = map(lambda leaf: FormulaTokenizer.leafToOPaths(leaf), leafs)
        upathLists = map(lambda leaf: FormulaTokenizer.leafToUPaths(leaf), leafs)
        sisters = FormulaTokenizer.findSisters(leafs)
        return [i for sublist in opathLists for i in sublist] + [i for sublist in upathLists for i in sublist] + sisters

    @classmethod
    def leafToOPaths(cls, leaf):
        opaths = []

        for i in range(len(leaf)):
            hierarchy = "#".join(map(lambda node: str(node.childCount), leaf[i:-1]))
            opaths.append("o:" + hierarchy + ("#" if len(hierarchy) != 0 else "") + leaf[-1].name + ("#" + leaf[-1].chars if len(leaf[-1].chars) != 0 else ""))

        return opaths

    @classmethod
    def leafToUPaths(cls, leaf):
        upaths = []

        for i in range(len(leaf)):
            hierarchy = (len(leaf)-(i+1)) * "#"
            upaths.append("u:" + hierarchy + leaf[-1].name + ("#" + leaf[-1].chars if len(leaf[-1].chars) != 0 else ""))

        return upaths

    @classmethod
    def findSisters(cls, leafs):
        sisters = []
        leafsAndParent = map(lambda leaf: [-1, leaf[-1]] if len(leaf) == 1 else [leaf[-2].id, leaf[-1]], leafs)

        for key, group in itertools.groupby(leafsAndParent, lambda x: x[0]):
            group = list(group)
            for i in range(0, len(group)-1):
                for j in range(i+1, len(group)):
                    s1 = group[i][1].name + ("#" + group[i][1].chars if len(group[i][1].chars) != 0 else "")
                    s2 = group[j][1].name + ("#" + group[j][1].chars if len(group[j][1].chars) != 0 else "")

                    sisters.append("s:" + s1 + "-" + s2)

        return sisters

    class Node(object):
        def __init__(self, name):
            self.name = name
            self.chars = ""
            self.childCount = 0
            self.id = FormulaTokenizer.nodeCounter
            FormulaTokenizer.nodeCounter += 1

    class FormulaContentHandler(xml.sax.ContentHandler):
        def __init__(self):
            xml.sax.ContentHandler.__init__(self)
            self.path = []
            self.tokenCandidateBuffer = []
            self.leafs = []
            self.irrelevantNodes = ['math', 'mrow', 'mfence']

        def startElement(self, name, attrs):
            if name in self.irrelevantNodes:
                return

            self.path.append(FormulaTokenizer.Node(name))
            if len(self.path) > 1:
                self.path[-2].childCount += 1

            self.tokenCandidateBuffer.append(copy.deepcopy(self.path))
            if len(self.tokenCandidateBuffer) > 3:
                self.tokenCandidateBuffer.pop(0)

        def endElement(self, name):
            if name in self.irrelevantNodes:
                return

            del self.path[-1]
            self.tokenCandidateBuffer.append(copy.deepcopy(self.path))
            if len(self.tokenCandidateBuffer) > 3:
                self.tokenCandidateBuffer.pop(0)

            if len(self.tokenCandidateBuffer) == 3:
                lengths = map(lambda x: len(x), self.tokenCandidateBuffer)
                if lengths[0] <= lengths[1] >= lengths[2]:
                    self.leafs.append(copy.deepcopy(self.tokenCandidateBuffer[1]))

        def characters(self, content):
            if len(content.strip()) == 0 or len(self.path) == 0:
                return

            currentLeafNode = self.path[-1]
            currentLeafNode.chars += content.strip()
            self.tokenCandidateBuffer[-1] = copy.deepcopy(self.path)


class TextTokenizer:
    def __init__(self):
        self.tokenizer = CountVectorizer(input='content', lowercase=True, stop_words='english', min_df=1).build_tokenizer()
        self.wnl = WordNetLemmatizer()

    def tokenize(self, text):
        return map(
            lambda x: "t:" + self.wnl.lemmatize(x.lower()),
            self.tokenizer(text)
        )


class MixedTextSeparator:
    def __init__(self):
        pass

    class FormulaId(object):
        def __init__(self, fid):
            self.fid = fid

        def __repr__(self):
            return str(self)

        def __str__(self):
            return "Formula(" + self.fid + ")"

    class TextPassage(object):
        def __init__(self, text):
            self.text = text

        def __repr__(self):
            return str(self)

        def __str__(self):
            return "Text(" + self.text + ")"

    def split(self, text):
        splitRe = re.compile("<fid (?P<fid>[^>]+)>")
        parts = []

        while True:
            m = splitRe.search(text)

            if m:
                parts.append(MixedTextSeparator.TextPassage(text[:m.start()]))
                parts.append(MixedTextSeparator.FormulaId(m.group('fid')))
                text = text[m.end():]
            else:
                break

        parts.append(MixedTextSeparator.TextPassage(text))
        return parts


# plotting
def hist(x, bounds):
    bins = {}

    for val in x:
        if val >= bounds[0]:
            for i in range(0, len(bounds)-1):
                if val < bounds[i+1]:
                    bins[i] = bins.get(i, 0) + 1
                    break

    labels = []
    for i in range(0, len(bounds)-1):
        lowerBound = bounds[i]
        upperBound = bounds[i+1] - 1

        if(lowerBound == upperBound):
            labels.append(str(lowerBound))
        else:
            labels.append(str(lowerBound) + " - " + str(upperBound))

    values = map(lambda x: x[1], sorted(bins.items(), key=lambda x: x[0]))

    return labels, values


def barPlot(plt, labels, valueLists, colors=['r', 'b', 'y', 'm'], width=0.6):
    numLists = len(valueLists)
    totalBars = len(valueLists[0])
    ind = np.arange(totalBars)

    rects = []
    ax = plt.axes()
    for values, i in zip(valueLists, range(len(valueLists))):
        r = ax.bar(left=ind+((0.5-(width/2)))+i*(width/numLists), height=values, width=width/numLists, color=colors[i % len(colors)])
        rects.append(r)

    ax.set_xticks(ind+0.5)
    ax.set_xticklabels(labels)

    ax.set_ylabel('#Categories')
    ax.set_xlabel('Best Jaccards (in %)')

    return rects
