from util import readFileLinewise2Array
from nltk.stem import WordNetLemmatizer
import re
from string import whitespace, ascii_uppercase

class zbMathTokenizer:
    def __init__(self):
        pass

    class WhiteSpace(object):
        def __init__(self):
            pass

        def strMapping(self):
            return " "

        def __str__(self):
            return "<WhiteSpace>"

        def __repr__(self):
            return str(self)

    class SeparatorChar(object):
        def __init__(self, c):
            self.c = c

        def strMapping(self):
            return self.c

        def __str__(self):
            return "<SeparatorChar " + self.c + ">"

        def __repr__(self):
            return str(self)

    class TokenChar(object):
        def __init__(self, c):
            self.c = c

        def strMapping(self):
            return self.c

        def __str__(self):
            return "<TokenChar " + self.c + ">"

        def __repr__(self):
            return str(self)

    class TokenString(object): 
        def __init__(self, str):
            self.str = str

        def strMapping(self):
            return self.str

        def __str__(self):
            return "<TokenString " + self.str + ">"

        def __repr__(self):
            return str(self)

    class Reference(object):
        def __init__(self, text):
            m = re.match(text, zbMathTokenizer.Reference.zblRe)
            if m:
                self.text = text
                self.nr = m.group(1)
            else:
                self.nr = None
                self.text = text

        def strMapping(self):
            if not self.nr is None:
                return "[" + self.nr + "]"
            else:
                return "[" + self.text + "]"

        def __str__(self):
            if not self.nr is None:
                return "<Reference " + self.nr + ">"
            else:
                return "<Reference " + self.text + ">"

        def __repr__(self):
            return str(self)

    Reference.zblRe = r"""([0-9]+\.[0-9]+)"""
        
    class Author(object):
        def __init__(self, text):
            self.text = text

        def split(self):
            parts = self.text.split(", ")
            return map(lambda p : zbMathTokenizer.Author(p), parts)

        def strMapping(self):
            return "{" + self.text + "}"

        def __str__(self):
            return "<Author " + self.text + ">"

        def __repr__(self):
            return str(self)

    class Formula(object):
        def __init__(self, text):
            self.text = text

        def strMapping(self):
            return "$" + self.text + "$"

        def __str__(self):
            return "<Formula " + self.text + ">"

        def __repr__(self):
            return str(self)
    
    class BracedText(object):
        def __init__(self, text):
            self.text = text

        def __str__(self):
            return "<BracedText " + self.text + ">"

        def __repr__(self):
            return str(self)
    
    # compress sequences of TokenCharacters to TokenStrings
    # leave rest as it is
    def compress(self, tokenList):
        tokenCharBuffer = []
        tokenBuffer = []
        for token in tokenList:
            if type(token) is zbMathTokenizer.TokenChar:
                tokenCharBuffer.append(token.c)
            else:
                if len(tokenCharBuffer) != 0:
                    tokenBuffer.append(
                        zbMathTokenizer.TokenString(''.join(tokenCharBuffer))
                    )
                    tokenCharBuffer = []
                tokenBuffer.append(token)

        if len(tokenCharBuffer) != 0:
            tokenBuffer.append(
                zbMathTokenizer.TokenString(''.join(tokenCharBuffer))
            )
        
        return tokenBuffer
    
    def tokenize(self, text, separators = ['.', ':', ';', ',']):
        characters = list(text)
        tokenList = []
        
        while(len(characters) > 0):
            tokenList.append(self.nextToken(characters, separators))
        
        return self.replaceAcronyms(self.splitMultipleAuthors(self.compress(tokenList)))

    def nextToken(self, characters, separators):
        if characters[0] == '$':
            try:
                return self.readFormula(characters)
            except:
                del characters[0]
                return zbMathTokenizer.TokenChar('$')
            
        elif characters[0] == '[':
            try:
                return self.readReference(characters)
            except:
                del characters[0]
                zbMathTokenizer.TokenChar('[')

        elif characters[0] == "{" and len(characters) > 1 and "".join(characters[1:5]) == "\\it ":
            return self.readAuthor(characters)
        
        elif characters[0] == '(':
            return self.readBracedText(characters)
        
        elif characters[0] in separators:
            c = characters[0]
            del characters[0]
            return zbMathTokenizer.SeparatorChar(c)

        elif characters[0] in whitespace:
            del characters[0]
            return zbMathTokenizer.WhiteSpace()

        else:
            c = characters[0]
            del characters[0]
            return zbMathTokenizer.TokenChar(c)
    
    def splitMultipleAuthors(self, tokens):
        newTokens = []
        for token in tokens:
            if type(token) is zbMathTokenizer.Author:
                authors = zbMathTokenizer.Author.split(token)
                newTokens.extend(authors)
            else:
                newTokens.append(token)

        return newTokens

    def replaceAcronyms(self, tokenList):
        acronyms = { }
        filteredTokenList = filter(lambda t: not type(t) is zbMathTokenizer.WhiteSpace, tokenList)

        i = 0
        for t in filteredTokenList:
            if (type(t) is zbMathTokenizer.BracedText 
                and len(t.text) > 1 and all(c in ascii_uppercase for c in t.text) 
                and i>=len(t.text)):

                candidates = filteredTokenList[i-len(t.text):i]
                
                candidatesMatch = True
                firstCharacters = []
                for candidate in candidates:
                    if not type(candidate) is zbMathTokenizer.TokenString:
                        candidatesMatch = False
                        break

                    firstCharacters.append(candidate.str[0].upper())

                candidatesMatch = candidatesMatch and t.text == "".join(firstCharacters)
                if candidatesMatch:
                    acronyms[t.text] = candidates
            i += 1
        
        newTokenList = []
        for token in tokenList:
            if type(token) is zbMathTokenizer.TokenString and token.str in acronyms:
                newTokenList.extend(acronyms[token.str]) 
            elif type(token) is zbMathTokenizer.BracedText:
                pass
            else:
                newTokenList.append(token)

        return newTokenList
        
    def readFormula(self, characters):
        if characters[0] != '$':
            raise ValueError("first char must be a dollar if readFormula funciton is called")
        
        i = 0
        iFrom = 0
        iTo = 0
        
        while i < len(characters) and characters[i] == '$':
            i += 1

        iFrom = i
        if iFrom > 2:
            raise ValueError("there were 3 dollars to introduce a formula...")
        
        while i < len(characters) and characters[i] != '$':
            i += 1
        iTo = i
        
        while i < len(characters) and characters[i] == '$':
            i += 1
        
        if iFrom != (i - iTo):
            raise ValueError("the number of opening and closing dollar signs do not match: " + text)
        
        formula = zbMathTokenizer.Formula("".join(characters[iFrom: iTo]))
        del characters[:i]
        return formula
    
    def readReference(self, characters):
        if characters[0] != '[':
            raise ValueError("first char must be a opening square bracket if readReference funciton is called")
        
        i = 1
        dept = 1
        while i < len(characters) and dept > 0:
            if characters[i] == ']':
                dept -= 1
            if characters[i] == '[':
                dept += 1
            i += 1
        
        sourceLink = zbMathTokenizer.Reference("".join(characters[1:i-1]))
        del characters[:i]
        return sourceLink

    def readAuthor(self, characters):
        if "".join(characters[0:5]) != "{\\it ":
            raise ValueError("an author link must start with \"{\\it \"")
        
        i = 4
        while i < len(characters) and characters[i] != '}':
            i += 1
        
        authorLink = zbMathTokenizer.Author("".join(characters[5:i]))
        del characters[:i+1]
        return authorLink
    
    def readBracedText(self, characters):
        if characters[0] != '(':
            raise ValueError("first char must be a opening brace if readBracedText funciton is called")
        
        i = 1
        while i < len(characters) and characters[i] != ')':
            i += 1
        
        bracedText = zbMathTokenizer.BracedText("".join(characters[1:i]))
        del characters[:i+1]
        return bracedText

zbMathTokenizer.stopList = readFileLinewise2Array("/home/simon/Projekte/MIRS/ClassificationFramework/util/stoplist.txt")
zbMathTokenizer.wnl = WordNetLemmatizer()

"""t = zbMathTokenizer()
print " ".join(map(lambda x : x.strMapping(), t.tokenize("this is a test (TIAT) $asdfasdf$ {\\it Grubel, Bert} TIAT is cool [zbl asdfasdf 1000.1000]")))"""