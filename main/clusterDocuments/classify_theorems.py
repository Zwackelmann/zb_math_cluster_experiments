from util import connectToDb, readFileLinewise2Array, build_csr_matrix, save_csr_matrix, load_csr_matrix
from util import indexes_in_list, groupAndCount
import random
import json
from sklearn import svm
from operator import itemgetter
import joblib
from sklearn.decomposition import TruncatedSVD

# dump msc_assignments to a file
"""db = connectToDb()
cursor = db.cursor()

cursor.execute("SELECT document, msc FROM msc_assignment WHERE pos=1")
l = []
for row in cursor:
    l.append((row[0], row[1]))

f = open("raw_data/doc2msc", "w")
for entry in l:
    f.write(entry[0] + ";" + entry[1] + "\n")
f.close()"""

# create id list for test and train set
"""f = open("raw_data/doc2msc")
random.seed(0)
testDocProportion = 0.3

trainDocIds = []
testDocIds = []
for line in f:
    parts = line.split(";")
    if random.random() < testDocProportion:
        testDocIds.append(parts[0])
    else:
        trainDocIds.append(parts[0])

def serialize(l, path):
    f = open(path, "w")
    for item in l:
        f.write(item + "\n")
    f.close()

serialize(trainDocIds, "raw_data/train_doc_ids")
serialize(testDocIds, "raw_data/test_doc_ids")"""

# group theorem matrix by documents
"""mat = load_csr_matrix("derived_data/tfidf_theorem_tdm.npz")
(num_rows, num_cols) = mat.shape

def aggregate_rows(mat, indexes):
    doc = mat[indexes, :].sum(axis=0).tolist()[0]
    doc_as_map = dict(filter(lambda z: z[1] != 0.0, zip(range(len(doc)), doc)))
    return doc_as_map

theorem_ids = json.load(open("derived_data/raw_theorem_tdm_theorem_ids"))
last_doc_id = None
collected_indexes = []
documents = []
index = 0
document_id_list = []
for theorem_id in theorem_ids:
    curr_doc_id = theorem_id[0]
    if curr_doc_id != last_doc_id:
        if last_doc_id is not None:
            print last_doc_id, len(document_id_list)
            document_id_list.append(last_doc_id)
            documents.append(aggregate_rows(mat, collected_indexes))
            collected_indexes = []
        last_doc_id = curr_doc_id

    collected_indexes.append(index)
    index += 1

if len(collected_indexes) != 0:
    document_id_list.append(last_doc_id)
    documents.append(aggregate_rows(mat, collected_indexes))

doc_mat = build_csr_matrix(documents, numAttributes=num_cols)
save_csr_matrix(doc_mat, "derived_data/tfidf_theorem_tdm_grouped_by_docs")
f = open("derived_data/theorem_tdm_grouped_by_docs_doc_ids", "w")
f.write(json.dumps(document_id_list))
f.close()"""

# cut test/train set out of corpus
document_ids = json.load(open("derived_data/theorem_tdm_grouped_by_docs_doc_ids"))
doc2msc = {}
f = open("raw_data/doc2msc")
for line in f:
    x = line.split(";")
    doc2msc[str(x[0])] = x[1].strip()
f.close()

target_class = "81"
ordered_document_assignments = map(lambda doc_id: doc2msc[str(doc_id)] if doc_id in doc2msc else None, document_ids)
ordered_document_labels = map(lambda lab: None if lab is None else (1 if lab[:len(target_class)] == target_class else 0), ordered_document_assignments)

test_doc_ind = indexes_in_list(document_ids, readFileLinewise2Array("raw_data/test_doc_ids"))
train_doc_ind = indexes_in_list(document_ids, readFileLinewise2Array("raw_data/train_doc_ids"))

mat = load_csr_matrix("derived_data/tfidf_theorem_tdm_grouped_by_docs.npz")
train_mat = mat[train_doc_ind, :]
train_labels = itemgetter(*train_doc_ind)(ordered_document_labels)

svd = TruncatedSVD(n_components=1000)
svd.fit(train_mat)

test_mat = mat[test_doc_ind, :]
test_labels = itemgetter(*test_doc_ind)(ordered_document_labels)

clf = svm.LinearSVC()
clf.fit(svd.transform(train_mat), train_labels)

# eval results
predictions = clf.predict(svd.transform(test_mat)).tolist()


def f(pred_label_pair):
    if pred_label_pair[0] == 1 and pred_label_pair[1] == 1:
        return "tp"
    elif pred_label_pair[0] == 1 and pred_label_pair[1] == 0:
        return "fp"
    elif pred_label_pair[0] == 0 and pred_label_pair[1] == 1:
        return "fn"
    else:
        return "tn"

cats = map(lambda x: f(x), zip(predictions, test_labels))
print groupAndCount(cats)
