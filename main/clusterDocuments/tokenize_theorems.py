from util import flatten, connect_to_db, get_all_document_ids, add_to_dict, FormulaTokenizer, MixedTextSeparator
from util import TextTokenizer, group_and_count, build_csr_matrix, save_csr_matrix, load_csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import json
import re
import joblib
from sklearn import svm
import random
from operator import itemgetter
import lda
import lda.datasets
import numpy as np
import itertools
import math
from fractions import Fraction
from gensim.models.ldamodel import LdaModel


def get_theorems_from_doc(document_id, cursor):
    cursor.execute("""
        SELECT paragraph_id, theorem_type, text FROM theorem
        WHERE document = %(document)s
    """, {"document": document_id})

    theorems = {}
    for row in cursor:
        theorems[row[0]] = {"type": row[1],
                            "text": row[2].decode('utf-16')}
    return theorems


def get_formulas_from_doc(document_id, cursor):
    cursor.execute("""
        SELECT formula_id, p_math_ml, c_math_ml FROM formula
        WHERE document = %(document)s
    """, {"document": document_id})

    formula_dict = {}
    for row in cursor:
        formula_dict[row[0]] = {"p_math_ml": row[1],
                                "c_math_ml": row[2]}

    return formula_dict


def tokenize_theorem(theorem_text, formula_dict, method):
    sep = MixedTextSeparator()
    ft = FormulaTokenizer()
    tt = TextTokenizer()

    tokens = []
    parts = sep.split(theorem_text)
    for part in parts:
        if type(part) is MixedTextSeparator.FormulaId:
            if part.fid in formula_dict:
                if method == "kristianto":
                    tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml'], method))
                elif method == "lin":
                    tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml'], method))
                else:
                    raise ValueError("Method must be either 'kristianto' or 'lin'")
            else:
                pass
        elif type(part) is MixedTextSeparator.TextPassage:
            tokens.extend(tt.tokenize(part.text))
        else:
            raise ValueError("Part is neither a FormulaId nor a TextPassage")

    return tokens


def get_theorem_tokens_from_doc(document_id, method, cursor):
    doc_theorems = get_theorems_from_doc(document_id, cursor)
    formula_dict = get_formulas_from_doc(document_id, cursor)

    theorem_token_list = []
    for theorem_id, theorem in doc_theorems.items():
        theorem_tokens = tokenize_theorem(theorem['text'], formula_dict, method)
        theorem_token_list.append(((document_id, theorem_id), theorem_tokens))

    return theorem_token_list


def get_all_theorems_as_token_list(method, cursor):
    document_ids = get_all_document_ids(cursor)
    return (theorem for document_id in document_ids for theorem in get_theorem_tokens_from_doc(document_id, method, cursor))


def calc_word_counts(items):
    global_token_counts = {}
    item_count = 1
    for id, tokens in items:
        print str(id) + " (" + str(item_count) + ")"
        add_to_dict(global_token_counts, group_and_count(tokens))
        item_count += 1

    return global_token_counts


def build_text_token_dict(global_feature_counts, min_count):
    frequent_tokens = map(lambda i: i[0], filter(lambda c: c[1] >= min_count, global_feature_counts.items()))
    token2index_map = dict(zip(sorted(frequent_tokens), range(len(frequent_tokens))))
    return token2index_map


def build_raw_csr_matrix(items, token2index_map):
    item_maps = []
    item_id_log = []

    item_count = 1
    for item_id, tokens in items:
        print str(item_id) + " (" + str(item_count) + ")"

        item_maps.append(group_and_count(tokens))
        item_id_log.append(item_id)
        item_count += 1

    m = build_csr_matrix(list_of_maps=item_maps, token_2_index_map=token2index_map)
    return m, item_id_log

if __name__ == "__main__":
    db = connect_to_db()
    cursor = db.cursor()
    token_method = "kristianto"

    # === calc word counts
    theorem_generator = get_all_theorems_as_token_list(token_method, cursor)
    token_counts = calc_word_counts(theorem_generator)

    f = open("derived_data/theorem_" + token_method + "token_counts.json", "w")
    f.write(json.dumps(token_counts))
    f.close()

    # === build text token dict
    token_counts = json.load(open("derived_data/theorem_" + token_method + "token_counts.json"))
    text_token_dict = build_text_token_dict(token_counts, 3)

    f = open("derived_data/theorem_" + token_method + "token2index_map.json", "w")
    f.write(json.dumps(text_token_dict))
    f.close()

    # === create raw csr_matrix for theorems
    token2index_map = json.load(open("derived_data/theorem_" + token_method + "token2index_map.json"))
    theorem_generator = get_all_theorems_as_token_list(token_method, cursor)

    matrix, id_log = build_raw_csr_matrix(theorem_generator, token2index_map)
    save_csr_matrix(matrix, "derived_data/theorem_" + token_method + "_raw_tdm")

    f = open("derived_data/theorem_" + token_method + "_raw_tdm_ids", "w")
    for theorem_id in id_log:
        f.write(theorem_id[0] + ";" + theorem_id[1] + "\n")
    f.close()

    # === train and dump tf-idf model for theorem texts
    """raw_theorem_tdm = load_csr_matrix("derived_data/theorem_lin_raw_tdm.npz")
    tfidf_trans = TfidfTransformer()
    tfidf_trans.fit(raw_theorem_tdm)

    joblib.dump(tfidf_trans, "models/lin_raw_theorem_tfidf_model")"""

    # === append author information to theorem ids
    """f = open("derived_data/theorem_raw_tdm_ids")
    f2 = open("derived_data/theorem_raw_tdm_ids_with_authors", "w")
    db = connect_to_db()
    cursor = db.cursor()
    findAuthorStmt = "SELECT display_name, zbmath_id FROM authorship WHERE document=%(document)s AND rank<=%(rank)s"

    for line in f:
        x = line.split(";")
        doc_id = x[0].strip()
        theorem_id = x[1].strip()

        authors = []
        cursor.execute(findAuthorStmt, {"document": doc_id, "rank": 2})
        for row in cursor:
            display_name = row[0]
            zbmath_id = row[1]
            authors.append((display_name, zbmath_id))

        newline = doc_id + ";" + theorem_id + ";" + ";".join(map(lambda author: author[0] + "(" + str(author[1]) + ")", authors)) + (";" * (2-len(authors)))
        f2.write(newline + "\n")

    f.close()
    f2.close()"""

    # === train lda
    """raw_theorem_tdm = load_csr_matrix("derived_data/theorem_lin_raw_tdm.npz")
    model = lda.LDA(n_topics=100, n_iter=500, random_state=1)
    model.fit(tfidf_trans.transform(raw_theorem_tdm))
    joblib.dump(model, "lin_topic_model")"""

    # print topic words
    """model = joblib.load("models/lin_topic_model")
    topic_word = model.topic_word_
    n_top_words = 20
    token2index_map = json.load(open("derived_data/theorem_lintoken2index_map.json"))
    vocab = map(lambda x: x[0], sorted(token2index_map.items(), key=lambda x: x[1]))

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print "Topic {}: {}".format(i, " ".join(map(lambda x: x.encode("utf-8").strip(), topic_words)))"""

    # transform matrix
    """raw_theorem_tdm = load_csr_matrix("derived_data/theorem_raw_tdm.npz")
    model = joblib.load("models/topic_model")
    for i in range(10):
        print model.doc_topic_[i, :].tolist()
        print ""
    """

    # === calc author probability
    """
    author_theorem_map = {}
    theorem_author_map = {}
    index = 0
    f = open("derived_data/theorem_raw_tdm_ids_with_authors")
    for line in f:
        x = line.split(";")
        document_id = x[0]
        theorem_id = x[1]

        authors = [x[2]]
        author2 = x[3].strip()
        if author2 != "":
            authors.append(author2)

        for author in authors:
            if author not in author_theorem_map:
                author_theorem_map[author] = []

            if index not in theorem_author_map:
                theorem_author_map[index] = []

            author_theorem_map[author].append((document_id, theorem_id, index))
            theorem_author_map[index].append(author)
        index += 1
    f.close()

    # print sorted(map(lambda x: (x[0], len(x[1])), author_theorem_map.items()), key=lambda x: x[1], reverse=True)[:20]
    author_query = "Paolo Lipparini(None)"
    # author_query = "Jnos Kollr(kollar.janos)"

    author_theorems = map(
        lambda thm: (thm[0], thm[1], thm[2], Fraction(1, len(author_theorem_map[author_query]))),
        author_theorem_map[author_query]
    )

    # model = joblib.load("models/topic_model")
    # theorem_topic_matrix = model.doc_topic_
    # joblib.dump(theorem_topic_matrix, "ttm")
    theorem_topic_matrix = joblib.load("ttm")
    num_theorems, num_topics = theorem_topic_matrix.shape

    # sum of all topic allocations is 1 for each theorem
    topics_from_theorems = {}
    for doc_id, theorem_id, index, theorem_prop in author_theorems:
        theorem_vector = theorem_topic_matrix[index,:].tolist()

        for topic_index in range(num_topics):
            topic_value = theorem_vector[topic_index]
            topic_prop = theorem_prop * Fraction(topic_value)
            if topic_index not in topics_from_theorems:
                topics_from_theorems[topic_index] = Fraction(0)

            topics_from_theorems[topic_index] += topic_prop

    theorems_from_topics = {}
    for topic_index in range(num_topics):
        print topic_index
        topic_prop = topics_from_theorems[topic_index]
        topic_vector = theorem_topic_matrix[:,topic_index].tolist()
        topic_sum = sum(topic_vector)

        for theorem_index in range(num_theorems):
            theorem_value = topic_vector[theorem_index]
            theorem_prop = topic_prop * (theorem_value/topic_sum)
            if theorem_index not in theorems_from_topics:
                theorems_from_topics[theorem_index] = 0

            theorems_from_topics[theorem_index] += theorem_prop

    x = Fraction(0)
    for index, prob in theorems_from_topics.items():
        x += prob

    print float(x)

    author_probs = {}
    for index, prob in theorems_from_topics.items():
        authors = theorem_author_map[index]
        for author in authors:
            if author not in author_probs:
                author_probs[author] = 0

            author_probs[author] += prob / len(authors)

    print sorted(author_probs.items(), key=lambda x: x[1], reverse=True)[:20]

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
