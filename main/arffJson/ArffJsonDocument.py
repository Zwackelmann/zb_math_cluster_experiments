from json import loads as json2py
from json import dumps as py2json

class ArffJsonDocument(object):
    def __init__(self, jsonObject):
        rawData = jsonObject[1]
        self.data = []
        
        if type(rawData) is list:
            self.storageType = ArffJsonDocument.dense
            self.data = rawData
        elif type(rawData) is dict:
            self.storageType = ArffJsonDocument.sparse

            for key, val in rawData.iteritems():
                self.data.append((int(key), val))
        else:
            raise ValueError('Invalid document data: ' + repr(rawData))

        rawMeta = jsonObject[0]
        self.id = rawMeta[0]
        self.classes = map(str, rawMeta[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]
        
    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        raise Exception("not implemented")

    def __str__(self):
        return "Document(id: " + self.id + ", classes: " + str(self.classes) + ", data: " + str(self.data) + ")"
    
    def toJson(self):
        if self.storageType == ArffJsonDocument.sparse:
            data = {}
            for dataItem in self.data:
                data[dataItem[0]] = dataItem[1]
        elif self.storageType == ArffJsonDocument.dense:
            data = self.data
        else:
            raise ValueError("Illegal value for storageType")

        return py2json([[self.id, self.classes], data])

ArffJsonDocument.dense = 0
ArffJsonDocument.sparse = 1

    
    