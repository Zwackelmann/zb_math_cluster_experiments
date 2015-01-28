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
# from gensim.models.ldamodel import LdaModel


def get_paragraphs_from_doc(document_id, data_basis, cursor):
    paragraphs = {}

    if data_basis == "only_theorems":
        cursor.execute("""
            SELECT paragraph_id, theorem_type, text FROM theorem
            WHERE document = %(document)s
        """, {"document": document_id})

        for row in cursor:
            paragraphs[row[0]] = {"type": row[1],
                                  "text": row[2].decode('utf-16')}
    elif data_basis == "full_text":
        cursor.execute("""
            SELECT paragraph_id, text FROM paragraph2
            WHERE document = %(document)s
        """, {"document": document_id})

        for row in cursor:
            paragraphs[row[0]] = {"text": row[1].decode('utf-16')}
    else:
        raise ValueError("data_basis must be either 'only_theorems' or 'full_text'")

    return paragraphs


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


def tokenize_paragraph(paragraph_text, formula_dict, method):
    sep = MixedTextSeparator()
    ft = FormulaTokenizer()
    tt = TextTokenizer()

    tokens = []
    parts = sep.split(paragraph_text)
    for part in parts:
        if type(part) is MixedTextSeparator.FormulaId:
            if part.fid in formula_dict:
                if method == "kristianto":
                    tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml'], method))
                elif method == "lin":
                    tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml'], method))
                elif method == "plaintext":
                    pass
                else:
                    raise ValueError("Method must be either 'kristianto' or 'lin'")
            else:
                pass
        elif type(part) is MixedTextSeparator.TextPassage:
            tokens.extend(tt.tokenize(part.text))
        else:
            raise ValueError("Part is neither a FormulaId nor a TextPassage")

    return tokens


def get_paragraph_tokens_from_doc(document_id, method, data_basis, cursor):
    doc_paragraphs = get_paragraphs_from_doc(document_id, data_basis, cursor)
    formula_dict = get_formulas_from_doc(document_id, cursor)

    paragraph_token_list = []
    for paragraph_id, paragraph in doc_paragraphs.items():
        paragraph_tokens = tokenize_paragraph(paragraph['text'], formula_dict, method)
        paragraph_token_list.append(((document_id, paragraph_id), paragraph_tokens))

    return paragraph_token_list


def get_all_paragraphs_as_token_list(method, data_basis, cursor, debug_max_items=None):
    document_ids = get_all_document_ids(cursor)

    gen = (paragraph for document_id in document_ids for paragraph in get_paragraph_tokens_from_doc(document_id, method, data_basis, cursor))
    if debug_max_items is None:
        return gen
    else:
        return itertools.islice(gen, 0, debug_max_items)


def get_all_documents_as_token_list(method, data_basis, cursor, debug_max_items=None):
    document_ids = get_all_document_ids(cursor)
    count = 0

    for document_id in document_ids:
        doc_tokens = {}
        paragraphs = get_paragraph_tokens_from_doc(document_id, method, data_basis, cursor)
        for id, tokens in paragraphs:
            add_to_dict(doc_tokens, group_and_count(tokens))

        if debug_max_items is not None and count >= debug_max_items:
            return

        count += 1

        yield (document_id, doc_tokens)


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


def setting_string(data_basis, token_method, granularity):
    return data_basis + "_" + token_method + "_" + granularity

if __name__ == "__main__":
    db = connect_to_db()
    cursor = db.cursor()

    interesting_settings = [
        # {"data_basis": "only_theorems", "token_method": "lin", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "lin", "granularity": "documents"},
        # {"data_basis": "only_theorems", "token_method": "kristianto", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "kristianto", "granularity": "documents"},
        # {"data_basis": "only_theorems", "token_method": "plaintext", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "plaintext", "granularity": "documents"},
        {"data_basis": "full_text", "token_method": "lin", "granularity": "documents"},
        # {"data_basis": "full_text", "token_method": "kristianto", "granularity": "documents"},
        {"data_basis": "full_text", "token_method": "plaintext", "granularity": "documents"}
    ]

    for setting in interesting_settings:
        print "setting: " + str(setting)
        data_basis = setting["data_basis"]
        token_method = setting["token_method"]
        granularity = setting["granularity"]

        debug_max_items = None

        if token_method == "lin" or token_method == "kristianto":
            intended_amount_of_formula_tokens = 10000
            intended_amount_of_text_tokens = 10000
        elif token_method == "plaintext":
            intended_amount_of_formula_tokens = None
            intended_amount_of_text_tokens = 20000
        else:
            raise ValueError("token_method must be either 'lin', 'kristianto' or 'plaintext'")

        # === calc word counts
        if granularity == "paragraphs":
            paragraph_generator = get_all_paragraphs_as_token_list(token_method, data_basis, cursor, debug_max_items=debug_max_items)
            token_counts = calc_word_counts(paragraph_generator)
        elif granularity == "documents":
            document_generator = get_all_documents_as_token_list(token_method, data_basis, cursor, debug_max_items=debug_max_items)
            token_counts = calc_word_counts(document_generator)
        else:
            raise ValueError("granularity must be either paragraphs or documents")

        f = open("derived_data/" + setting_string(data_basis, token_method, granularity) + "__token_counts.json", "w")
        f.write(json.dumps(token_counts))
        f.close()

        # === build text token dict
        token_counts = json.load(open("derived_data/" + setting_string(data_basis, token_method, granularity) + "__token_counts.json"))
        text_token_dict = build_text_token_dict(token_counts, 3)

        f = open("derived_data/" + setting_string(data_basis, token_method, granularity) + "__token2index_map.json", "w")
        f.write(json.dumps(text_token_dict))
        f.close()

        # === create raw csr_matrix for theorems
        token2index_map = json.load(open("derived_data/" + setting_string(data_basis, token_method, granularity) + "__token2index_map.json"))
        if granularity == "paragraphs":
            paragraph_generator = get_all_paragraphs_as_token_list(token_method, data_basis, cursor, debug_max_items=debug_max_items)
            matrix, id_log = build_raw_csr_matrix(paragraph_generator, token2index_map)
        elif granularity == "documents":
            document_generator = get_all_documents_as_token_list(token_method, data_basis, cursor, debug_max_items=debug_max_items)
            matrix, id_log = build_raw_csr_matrix(document_generator, token2index_map)
        else:
            raise ValueError("granularity must be either paragraphs or documents")

        save_csr_matrix(matrix, "derived_data/" + setting_string(data_basis, token_method, granularity) + "__raw_tdm")

        f = open("derived_data/" + setting_string(data_basis, token_method, granularity) + "__ids", "w")
        if granularity == "paragraphs":
            for id in id_log:
                f.write(id[0] + ";" + id[1] + "\n")
        elif granularity == "documents":
            for id in id_log:
                f.write(id)
        f.close()

        # === train and dump tf-idf model for theorem texts
        # raw_tdm = load_csr_matrix("derived_data/" + setting_string(data_basis, token_method, granularity) + "__raw_tdm.npz")
        # tfidf_trans = TfidfTransformer()
        # tfidf_trans.fit(raw_tdm)

        # joblib.dump(tfidf_trans, "models/" + setting_string(data_basis, token_method, granularity) + "__tfidf_model")

        # === retrieve best tf-idf terms
        # raw_tdm = load_csr_matrix("derived_data/" + setting_string(data_basis, token_method, granularity) + "__raw_tdm.npz")
        # tfidf_trans = joblib.load("models/" + setting_string(data_basis, token_method, granularity) + "__tfidf_model")

        # vocab = json.load(open("derived_data/" + setting_string(data_basis, token_method, granularity) + "__token2index_map.json"))
        # text_token_indexes = sorted(map(lambda i: i[1], filter(lambda i: i[0][:2] == "t:", vocab.items())))
        # formula_token_indexes = sorted(map(lambda i: i[1], filter(lambda i: i[0][:2] != "t:", vocab.items())))

        # tfidf_tdm = tfidf_trans.transform(raw_tdm)
        # token_scores = list(enumerate(tfidf_tdm.sum(axis=0).tolist()[0]))

        # text_token_scores = sorted(itemgetter(*text_token_indexes)(token_scores), key=lambda x: x[1], reverse=True)
        # formula_token_scores = sorted(itemgetter(*formula_token_indexes)(token_scores), key=lambda x: x[1], reverse=True)

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
