from util import flatten, connect_to_db, get_all_document_ids, add_to_dict, FormulaTokenizer, MixedTextSeparator
from util import TextTokenizer, group_and_count, build_csr_matrix, save_csr_matrix, load_csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import json
import re
import joblib
from sklearn import svm
import random
from operator import itemgetter

testTheorem = """Let Assumption 1.4  be satisfied. Fix  <fid Thmtheorem1.p1.1.1.m1.1> , such that
<fid Thmtheorem1.p1.1.1.m2.1> . Let  <fid Thmtheorem1.p1.1.1.m3.1> , where  <fid Thmtheorem1.p1.1.1.m4.1> .
The nonlinear elliptic problem ( 1.3 ) with  <fid Thmtheorem1.p1.1.1.m5.1>  has a non-trivial solution
<fid Thmtheorem1.p1.1.1.m6.1>  in the form ( 1.10 ) with  <fid Thmtheorem1.p1.1.1.m7.1>  for any
<fid Thmtheorem1.p1.1.1.m8.1>  and sufficiently small  <fid Thmtheorem1.p1.1.1.m9.1>  if and only if
there exists a non-trivial solution for  <fid Thmtheorem1.p1.1.1.m10.1>  of the bifurcation equations
<fid S1.E12.m1.1> (1.12) where  <fid Thmtheorem1.p1.2.1.m1.1>  and  <fid Thmtheorem1.p1.2.1.m2.1>  are
analytic functions of  <fid Thmtheorem1.p1.2.1.m3.1>  near  <fid Thmtheorem1.p1.2.1.m4.1>  satisfying
the bounds <fid S1.E13.m1.1> (1.13) for sufficiently small  <fid Thmtheorem1.p1.3.1.m1.1> , fixe
<fid Thmtheorem1.p1.3.1.m2.1> , and some constants  <fid Thmtheorem1.p1.3.1.m3.1>  which are
independent of  <fid Thmtheorem1.p1.3.1.m4.1> and depend on  <fid Thmtheorem1.p1.3.1.m5.1> . Moreover
<fid Thmtheorem1.p1.3.1.m6.1> ,  <fid Thmtheorem1.p1.3.1.m7.1>  and <fid S1.E14.m1.1> (1.14) for som
<fid Thmtheorem1.p1.4.1.m1.1> -independent constant  <fid Thmtheorem1.p1.4.1.m2.1> ."""


def get_theorem_feature_counts(document_id, cursor):
    cursor.execute("""
        SELECT paragraph_id, theorem_type, text FROM theorem
        WHERE document = %(document)s
    """, {"document": document_id})

    theorems = {}
    for row in cursor:
        theorems[row[0]] = {"type": row[1],
                            "text": row[2].decode('utf-16')}

    cursor.execute("""
        SELECT formula_id, c_math_ml FROM formula
        WHERE document = %(document)s
    """, {"document": document_id})

    formula_dict = {}
    for row in cursor:
        formula_dict[row[0]] = {"c_math_ml": row[1]}

    theorem_feature_maps = []
    for theorem_id, theorem in theorems.items():
        theorem_tokens = tokenize_theorem(theorem['text'], formula_dict)
        feature_map = group_and_count(theorem_tokens)
        theorem_feature_maps.append((theorem_id, feature_map))

    return theorem_feature_maps


def tokenize_theorem(theorem_text, formula_dict):
    sep = MixedTextSeparator()
    ft = FormulaTokenizer()
    tt = TextTokenizer()

    tokens = []
    parts = sep.split(theorem_text)
    for part in parts:
        if type(part) is MixedTextSeparator.FormulaId:
            if part.fid in formula_dict:
                tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml']))
            else:
                pass
        elif type(part) is MixedTextSeparator.TextPassage:
            tokens.extend(tt.tokenize(part.text))
        else:
            raise ValueError("Part is neither a FormulaId nor a TextPassage")

    return tokens


"""def tokenizeFormula(formula):
    f = FormulaTokenizer()
    return f.parse(formula)"""


"""def theoremsToFeatureCounts(theorems):
    textTokenList = filter(lambda token: not(token[:5] == "<fid "), flatten(flatten(theorems)))

    tokenCounts = {}
    for token in textTokenList:
        if token not in tokenCounts:
            tokenCounts[token] = 0
        tokenCounts[token] = tokenCounts[token] + 1

    return tokenCounts"""

if __name__ == "__main__":
    # === calc word counts
    """db = connect_to_db()
    cursor = db.cursor()
    document_ids = get_all_document_ids(cursor)

    global_token_counts = {}
    doc_count = 1
    for document_id in document_ids:
        print document_id + " (" + str(doc_count) + "/" + str(len(document_ids)) + ")"
        theorem_feature_counts = get_theorem_feature_counts(document_id, cursor)
        for c in theorem_feature_counts:
            add_to_dict(global_token_counts, c[1])

        doc_count += 1

    f = open("derived_data/theorem_gtoken_counts.json", "w")
    f.write(json.dumps(global_token_counts))
    f.close()

    db.close()"""

    # === build text token dict
    """tokenCounts = json.load(open("derived_data/theorem_token_counts.json"))
    frequentTokens = map(lambda i: i[0], filter(lambda c: c[1] >= 10, tokenCounts.items()))
    token2IndexMap = dict(zip(sorted(frequentTokens), range(len(frequentTokens))))

    f = open("derived_data/theorem_token2index_map.json", "w")
    f.write(json.dumps(token2IndexMap))
    f.close()"""

    # === create raw csr_matrix for theorem texts
    """db = connect_to_db()
    cursor = db.cursor()
    document_ids = get_all_document_ids(cursor)

    token_2_index_map = json.load(open("derived_data/theorem_token2index_map.json"))

    document_maps = []
    document_id_log = []

    doc_count = 1
    for document_id in document_ids:
        print document_id + " (" + str(doc_count) + "/" + str(len(document_ids)) + ")"
        theorem_dict = {}
        theorem_feature_counts = get_theorem_feature_counts(document_id, cursor)
        for c in theorem_feature_counts:
            add_to_dict(theorem_dict, c[1])

        if len(theorem_dict) != 0:
            document_maps.append(theorem_dict)
            document_id_log.append(document_id)
        doc_count += 1

    m = build_csr_matrix(list_of_maps=document_maps, token_2_index_map=token_2_index_map)
    save_csr_matrix(m, "derived_data/theorem_raw_tdm_by_doc")
    f = open("derived_data/theorem_raw_tdm_by_doc_ids", "w")
    for doc_id in document_id_log:
        f.write(doc_id + "\n")
    f.close()
    """

    # === train and dump tf-idf model for theorem texts
    """raw_theorem_tdm = load_csr_matrix("derived_data/theorem_raw_tdm_by_doc.npz")
    tfidf_trans = TfidfTransformer()
    tfidf_trans.fit(raw_theorem_tdm)

    joblib.dump(tfidf_trans, "models/raw_theorem_by_doc_tfidf_model")
    """

    # === train svm
    """
    # read matrix
    raw_theorem_tdm = load_csr_matrix("derived_data/theorem_raw_tdm_by_doc.npz")

    # read doc2msc map
    f = open("raw_data/doc2msc", "r")
    doc2msc_map = {}
    for line in f:
        x = line.split(";")
        doc2msc_map[x[0].strip()] = x[1].strip()
    f.close()

    # read document ids
    f = open("derived_data/theorem_raw_tdm_by_doc_ids")
    doc_ids = []
    for line in f:
        x = line.strip()
        doc_ids.append(x)

    # create label map
    target_class = "81"
    label_vector = map(lambda id: None if id not in doc2msc_map else (1 if doc2msc_map[id][:2] == target_class else 0), doc_ids)

    # determine train and test sets
    test_indexes = []
    train_indexes = []
    random.seed(0)

    count = 0
    for doc_id in doc_ids:
        if doc_id in doc2msc_map:
            if random.random() > 0.2:
                train_indexes.append(count)
            else:
                test_indexes.append(count)

        count += 1

    print test_indexes

    train_matrix = raw_theorem_tdm[train_indexes, :]
    test_matrix = raw_theorem_tdm[test_indexes, :]
    train_labels = itemgetter(*train_indexes)(label_vector)
    test_labels = itemgetter(*test_indexes)(label_vector)

    # train classifier
    clf = svm.LinearSVC()
    clf.fit(train_matrix, train_labels)
    joblib.dump(clf, "models/theorem_raw_tdm_by_doc_svm_model")

    # predict test instances
    clf = joblib.load("models/theorem_raw_tdm_by_doc_svm_model")
    predictions = clf.predict(test_matrix).tolist()

    # evaluate
    evaluation_classes = map(lambda x: "tp" if x[0] == 1 and x[1] == 1 else ("fp" if x[0] == 1 and x[1] == 0 else ("fn" if x[0] == 0 and x[1] == 1 else "tn")), zip(predictions, test_labels))
    grouped_evaluation_classes = group_and_count(evaluation_classes)

    print "precition: " + str(float(grouped_evaluation_classes["tp"])/(grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fp"]))
    print "recall: " + str(float(grouped_evaluation_classes["tp"])/(grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fn"]))"""
