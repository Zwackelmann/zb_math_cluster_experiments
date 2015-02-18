import httplib
import urllib
import json
from pprint import pprint
from main.clusterDocuments.util import get_filenames_and_filepaths, files_in_dict, DocumentParser
from string import printable
import time
from os import path
from socket import error as SocketError
import errno

api_key1 = "a1036f9334647a67fefa01d78c159311"
api_key2 = "893a8d2b5dc7d5923e86a25b0f17dc01"
api_key3 = "6492f9c867ddf3e84baa10b5971e3e3d"
api_key = api_key3

elsevier_url = "api.elsevier.com:80"

conn = None


def get_connection():
    global conn
    if conn == None:
        conn = httplib.HTTPConnection(elsevier_url)
    
    return conn


def refresh_connection():
    global conn
    conn = httplib.HTTPConnection(elsevier_url)


def get_doi_data(doi):
    while True:
        try:
            get_connection().request(
                "GET",
                "/content/abstract/doi/%(doi)s?apiKey=%(api-key)s" % {
                    "doi": doi,
                    "api-key": api_key}
            )
            res = get_connection().getresponse()
            return res.status, res.read()
        except SocketError as e:
            if e.errno == errno.ECONNRESET:
                refresh_connection()
            else:
                raise e


p = DocumentParser()
for filename, filepath in zip(*get_filenames_and_filepaths("raw_data/ntcir_filenames")):
    print "parsing document " + filename
    document = p.parse_metadata(filepath)
    dois = map(lambda x: x.ident, filter(lambda x: x.source == "doi", document.identifiers))

    if len(dois) != 0:
        doi = dois[0]

        save_filename = filter(lambda c: c in printable, doi.replace('/', '_'))

        if not path.isfile("downloaded_abstract_data/" + save_filename + ".xml"):
            while True:
                status, text = get_doi_data(doi)
                if status == 200:
                    with open("downloaded_abstract_data/" + save_filename + ".xml", "w") as f:
                        f.write(text)
                    time.sleep(5)
                    print "success"
                    break
                elif status == 429:
                    print "too many requests. Wait 2 hours"
                    time.sleep(60*60*2)
                elif status == 404:
                    with open("downloaded_abstract_data/" + save_filename + ".xml", "w") as f:
                        f.write("not found")
                    print "not found"
                    time.sleep(5)
                    break
                else:
                    print "status " + str(status)
                    time.sleep(5)
                    break
	else:
            print "already exists"
    else:
        print "no doi available"
