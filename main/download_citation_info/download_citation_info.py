import httplib
import urllib
import json
from pprint import pprint
from main.clusterDocuments.util import get_filenames_and_filepaths, files_in_dict, DocumentParser
from string import printable

conn = httplib.HTTPConnection("api.elsevier.com:80")

api_key1 = "a1036f9334647a67fefa01d78c159311"
api_key2 = "893a8d2b5dc7d5923e86a25b0f17dc01"
api_key3 = "6492f9c867ddf3e84baa10b5971e3e3d"
api_key = api_key3

headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/xml"}
doi = "10.1016/j.disc.200710.025"
doi2 = "10.1016/S0014-5793(01)03313-0"
doi3 = "10.1016/j.disc.2007.10.025"
doi4 = "10.1007/s00220-008-0461-1"


def get_doi_data(doi):
    conn.request(
        "GET",
        "/content/abstract/doi/%(doi)s?apiKey=%(api-key)s" % {
            "doi": doi,
            "api-key": api_key}
    )
    res = conn.getresponse()

    print res.status
    return res.read()

filenames = files_in_dict("raw_data/test_documents", with_path=False)
p = DocumentParser()

for filename, filepath in get_filenames_and_filepaths():
    print "parsing document " + filename
    document = p.parse_metadata(filepath)
    dois = map(lambda x: x.ident, filter(lambda x: x.source == "doi", document.identifiers))

    if len(dois) != 0:
        doi = dois[0]

        save_filename = filter(lambda c: c in printable, doi.replace('/', '_'))

        with open("downloaded_abstract_data/" + save_filename + ".xml", "w") as f:
            f.write(get_doi_data(doi))
