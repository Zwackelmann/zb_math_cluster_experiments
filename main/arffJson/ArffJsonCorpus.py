from json import loads as json2py
from ArffJsonDocument import ArffJsonDocument
import numpy as np
from scipy.sparse import csr_matrix

class ArffJsonCorpus:
    def __init__(self, filename):
        self.filename = filename

        file = open(self.filename)
        self.header = json2py(file.next())
        file.close()
    
    def __iter__(self):
        file = open(self.filename)
        file.next() # skip header

        for line in file:
            yield ArffJsonDocument(json2py(line))

    def toNumpyArray(self):
        return np.array(list(self))

    def toCsrMatrix(self, projection = None, shapeCols = None, selection = None):
        row = []
        col = []
        data = []

        if projection is None:
            numDocs = 0
            for doc in iter(self):
                if selection is None or selection(doc):
                    numDocs += 1
                    for key, val in doc.data:
                        row.append(numDocs-1)
                        col.append(key)
                        data.append(val)

            shapeRows = numDocs

            if shapeCols is None:
                shapeCols = max(col)+1

            return csr_matrix( (data,(row,col)), shape=(shapeRows, shapeCols) )
        else:
            index2ProjectIndex = dict(zip(projection, range(len(projection))))

            numDocs = 0
            for doc in iter(self):
                if selection is None or selection(doc):
                    numDocs += 1

                    for key, val in doc.data:
                        if key in index2ProjectIndex:
                            row.append(numDocs-1)
                            col.append(index2ProjectIndex[key])
                            data.append(val)
            
            shapeRows = numDocs

            if shapeCols is None:
                shapeCols = max(col)+1

            return csr_matrix( (data,(row,col)), shape=(shapeRows, shapeCols) ), index2ProjectIndex
            