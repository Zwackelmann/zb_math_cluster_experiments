from string import ascii_letters, digits, printable
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import re
import json
from util import addToDict, groupAndCount, build_csr_matrix, save_csr_matrix, load_csr_matrix
import sys
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GMM
from zbMathTokenizer import zbMathTokenizer

class MixedTokenizer:
    def __init__(self):
        self.textTokenizer = TextTokenizer()
        self.formulaTokenizer = FormulaTokenizer()

    def tokenize(self, text):
        documentParts = self.distinguishDocumentParts(text)
        tokens = []
        for part in documentParts:
            if type(part) is MixedTokenizer.TextContent:
                tokens.extend(self.textTokenizer.tokenize(part.content))
            elif type(part) is MixedTokenizer.FormulaContent:
                tokens.extend(self.formulaTokenizer.tokenize(part.content))
            else:
                raise ValueError(str(type(part)) + " is not supported")

        return tokens

    def distinguishDocumentParts(self, text):
        documentParts = []
        remainingText = text[:]

        while True:
            match = re.search(r"\$[^\$]*\$", remainingText)

            if match:
                documentParts.append(MixedTokenizer.TextContent(remainingText[:match.start()]))
                documentParts.append(MixedTokenizer.FormulaContent(remainingText[match.start()+1: match.end()-1]))
                remainingText = remainingText[match.end():]
            else:
                break

        if len(remainingText) != 0:
            documentParts.append(MixedTokenizer.TextContent(remainingText))

        return documentParts

    class TextContent(object):
        def __init__(self, content):
            self.content = content

        def __repr__(self):
            return "TextContent(" + self.content + ")"

    class FormulaContent(object):
        def __init__(self, content):
            self.content = content

        def __repr__(self):
            return "FormulaContent(" + self.content + ")"

class TextTokenizer:
    def __init__(self):
        self.tokenizer = CountVectorizer(input='content', lowercase=True, min_df=1).build_tokenizer()
        self.wnl = WordNetLemmatizer()

    def tokenize(self, text):
        return filter(lambda t: t not in TextTokenizer.stopwords, map(
            lambda x: self.wnl.lemmatize(x.lower()),
            self.tokenizer(text)
        ))

TextTokenizer.stopwords = [
    "the", "of", "and", "in", "is", "to", "for", "with", "that", "are", "on", "we", "it", 
    "this", "by", "an", "be", "a", "problem", "model", "method", "result", "which", 
    "paper", "par", "solution", "author", "can", "some", "if", "such", "also", "at", 
    "using", "or", "show", "these", "where", "then", "given", "zbl" # "zbl" is used when referencing other documents (e.g. 1994; Zbl 0818.30023)
    ] # extracted from the corpus
    
class FormulaTokenizer:
    def __init__(self):
        pass

    def tokenize(self, matStr):
        if matStr is None or len(matStr) == 0:
            return []
        else:
            """tokens = []
            tokenBuffer = None
            strlen = len(matStr)
            state = 0
            index = 0

            while strlen>index:
                char = matStr[index]
                if state==0:
                    if char == '\\':
                        state = 1
                    elif char in FormulaTokenizer.letters_and_special_chars:
                        tokens.append("$" + char + "$")
                    else:
                        pass
                        
                elif state==1:
                    if char in ascii_letters:
                        tokenBuffer = char
                        state=2
                    else:
                        tokens.append("$" + char + "$")
                        state=0
                elif state==2:
                    if char in ascii_letters:
                        tokenBuffer += char
                    else:
                        tokens.append("$" + tokenBuffer + "$")
                        tokenBuffer = None
                        index-=1
                        state=0
                else:
                    raise ValueError("Undefined state while tokenizing")
                index+=1

            if tokenBuffer != None:
                tokens.append("$" + tokenBuffer + "$")

            return tokens"""

            return [filter(lambda c: c in digits+ascii_letters, matStr)]

FormulaTokenizer.consciously_ignored = '{},. %\n:~;$&?`"?@()_'
FormulaTokenizer.valid_special_chars = '+-*/|[]!<>=^'
FormulaTokenizer.letters_and_special_chars = ascii_letters + FormulaTokenizer.valid_special_chars + digits

def readDict(dictFilepath):
    lines = [line.strip() for line in open(dictFilepath)]
    return dict(zip(lines, xrange(len(lines))))

def validChar(char):
    return ord(char) >= 32 and char in printable

def line2Document(line):
    line = filter(lambda c: validChar(c), line)
    return json.loads(line)

def clusterDocument(title, abstract, tokenizer, token2indexMap, tfidf_model, lsi_model, gmm_model):
    tokens = tokenize(title + " " + abstract, tokenizer)
    tokenCounts = groupAndCount(tokens)
    matrix = build_csr_matrix(listOfMaps=[tokenCounts], token2IndexMap=token2indexMap)
    transformedMatrix = lsi_model.transform(tfidf_model.transform(matrix))
    prediction = gmm_model.predict(transformedMatrix)[0]
    return prediction

def tokenize(text, tokenizer):
    tokens = tokenizer.tokenize(text)

    strTokens = []
    for token in tokens:
        if type(token) == zbMathTokenizer.TokenString:
            rawToken = filter(lambda c: c in ascii_letters, token.str)
            if not rawToken in zbMathTokenizer.stopList:
                strTokens.append(transformStrings(rawToken))
        elif type(token) == zbMathTokenizer.TokenChar:
            strTokens.append(transformStrings(token.c))
        elif type(token) == zbMathTokenizer.Formula:
            strTokens.append( "$" + filter(lambda c: c in ascii_letters+digits, token.text) + "$")
        elif type(token) == zbMathTokenizer.Author:
            strTokens.append( "{" + token.text + "}")

    return strTokens

stemmer = PorterStemmer()
def transformStrings(x):
    global stemmer
    return stemmer.stem(x.lower())

suffix = "zbl_tokenizer"
# calc token counts
# t = MixedTokenizer()
"""t = zbMathTokenizer()

globalTokenCounts = { }
for line in open("raw_data/simple_corpus.json"):
    document = line2Document(line)
    tokens = tokenize(document[1] + " " + document[2], t)

    tokenCounts = groupAndCount(tokens)
    addToDict(globalTokenCounts, tokenCounts)

f = open("derived_data/simple_corpus_" + suffix + "_token_counts.json", "w")
f.write(json.dumps(globalTokenCounts))
f.close()"""

# build token dict
"""minTokenCount = 10

tokenCounts = json.load(open("derived_data/simple_corpus_" + suffix + "_token_counts.json"))
frequentTokens = map(lambda i: i[0], filter(lambda c : c[1] >= minTokenCount, tokenCounts.items()))

token2IndexMap = dict(zip(sorted(frequentTokens), range(len(frequentTokens))))

f = open("derived_data/simple_corpus_" + suffix + "_token2index_map.json", "w")
f.write(json.dumps(token2IndexMap))
f.close()"""

# create raw csr_matrix
"""t = zbMathTokenizer()
token2IndexMap = json.load(open("derived_data/simple_corpus_" + suffix + "_token2index_map.json"))

l = []
for line in open("raw_data/simple_corpus.json"):
    document = line2Document(line)
    tokens = tokenize(document[1] + " " + document[2], t)
    tokenCounts = groupAndCount(tokens)
    l.append(tokenCounts)

m = build_csr_matrix(listOfMaps=l, token2IndexMap=token2IndexMap)
save_csr_matrix(m, "derived_data/simple_corpus_" + suffix + "_raw_tdm")"""

# gen tf-idf model
"""simple_corpus_tdm = load_csr_matrix("derived_data/simple_corpus_" + suffix + "_raw_tdm.npz")
tfidf_model = TfidfTransformer()
tfidf_model.fit(simple_corpus_tdm)

joblib.dump(tfidf_model, "models/simple_corpus_" + suffix + "_tfidf_model")"""

# gen lsa model
"""simple_corpus_tdm = load_csr_matrix("derived_data/simple_corpus_" + suffix + "_raw_tdm.npz")
tfidf_model = joblib.load("models/simple_corpus_" + suffix + "_tfidf_model")
simple_corpus_tdm = tfidf_model.transform(simple_corpus_tdm)

svd = TruncatedSVD(n_components=250)
svd.fit(simple_corpus_tdm)
joblib.dump(svd, "models/simple_corpus_" + suffix + "_tfidf-lsi_model")"""

# gen cluster model
"""simple_corpus_tdm = load_csr_matrix("derived_data/simple_corpus_" + suffix + "_raw_tdm.npz")
tfidf_model = joblib.load("models/simple_corpus_" + suffix + "_tfidf_model")
lsi_model = joblib.load("models/simple_corpus_" + suffix + "_tfidf-lsi_model")

gmm_model = GMM(n_components=64)
gmm_model.fit(lsi_model.transform(tfidf_model.transform(simple_corpus_tdm)))
joblib.dump(gmm_model, "models/gmm-simple_corpus_" + suffix + "_tfidf-lsi")"""

# cluster documents
tokenizer = zbMathTokenizer()
token2IndexMap = json.load(open("derived_data/simple_corpus_" + suffix + "_token2index_map.json"))
tfidf_model = joblib.load("models/simple_corpus_" + suffix + "_tfidf_model")
lsi_model = joblib.load("models/simple_corpus_" + suffix + "_tfidf-lsi_model")
gmm_model = joblib.load("models/gmm-simple_corpus_" + suffix + "_tfidf-lsi")

f = open("simple_corpus_" + suffix + "_gmm_clusters", "w")
count = 0
skip = 3
take = 1
for line in open("raw_data/simple_corpus.json"):
    if count >= skip:
        document = line2Document(line)
        id = document[0]
        title = document[1]
        abstract = document[2]
        prediction = clusterDocument(title, abstract, tokenizer, token2IndexMap, tfidf_model, lsi_model, gmm_model)
        f.write(id + ";" + str(prediction) + "\n")

    #if count == skip+take-1:
    #    break

    count += 1
f.close()

