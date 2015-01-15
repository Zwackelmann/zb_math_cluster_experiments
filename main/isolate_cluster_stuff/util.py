from string import ascii_letters, digits, printable
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import json

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

            return tokens

FormulaTokenizer.consciously_ignored = '{},. %\n:~;$&?`"?@'
FormulaTokenizer.valid_special_chars = '+-*/_|[]()!<>=^'
FormulaTokenizer.letters_and_special_chars = ascii_letters + FormulaTokenizer.valid_special_chars + digits

def readDict(dictFilepath):
    lines = [line.strip() for line in open(dictFilepath)]
    return dict(zip(lines, xrange(len(lines))))

t = MixedTokenizer()

print token2Index
