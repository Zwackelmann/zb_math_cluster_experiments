from util import flatten, connect_to_db, get_all_document_ids, add_to_dict, FormulaTokenizer, MixedTextSeparator
from util import TextTokenizer, group_and_count, build_csr_matrix, save_csr_matrix, load_csr_matrix
from util import vertically_append_matrix
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
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, Normalizer

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


def tokenize_paragraph(paragraph_text, formula_dict, method, config_args):
    sep = MixedTextSeparator()
    ft = FormulaTokenizer()
    tt = TextTokenizer()

    tokens = []
    parts = sep.split(paragraph_text)
    for part in parts:
        if type(part) is MixedTextSeparator.FormulaId:
            if part.fid in formula_dict:
                if method == "kristianto":
                    tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml'], method, config_args))
                elif method == "lin":
                    tokens.extend(ft.tokenize(formula_dict[part.fid]['c_math_ml'], method, config_args))
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


def get_paragraph_tokens_from_doc(document_id, method, data_basis, cursor, config_args={}):
    doc_paragraphs = get_paragraphs_from_doc(document_id, data_basis, cursor)
    formula_dict = get_formulas_from_doc(document_id, cursor)

    paragraph_token_list = []
    for paragraph_id, paragraph in doc_paragraphs.items():
        paragraph_tokens = tokenize_paragraph(paragraph['text'], formula_dict, method, config_args)
        paragraph_token_list.append(((document_id, paragraph_id), paragraph_tokens))

    return paragraph_token_list


def get_all_paragraphs_as_token_list(method, data_basis, cursor, config_args={}):
    document_ids = get_all_document_ids(cursor)

    gen = (paragraph for document_id in document_ids for paragraph in get_paragraph_tokens_from_doc(document_id, method, data_basis, cursor, config_args=config_args))
    if config_args.get("debug_max_items") is None:
        return gen
    else:
        return itertools.islice(gen, 0, config_args["debug_max_items"])


def get_all_documents_as_token_list(method, data_basis, cursor, config_args={}):
    document_ids = get_all_document_ids(cursor)
    count = 0

    for document_id in document_ids:
        doc_tokens = {}
        paragraphs = get_paragraph_tokens_from_doc(document_id, method, data_basis, cursor, config_args=config_args)
        for id, tokens in paragraphs:
            add_to_dict(doc_tokens, group_and_count(tokens))

        if config_args.get("debug_max_items") is not None and count >= config_args["debug_max_items"]:
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


def matrix_prune(mat, factor):
    newdata = np.array(map(lambda x: float(x)*factor, mat.data))
    return csr_matrix((newdata, mat.indices, mat.indptr), shape=mat.shape)


def row_wise_norm(mat):
    matcopy = mat.copy()
    matcopy.data **= 2
    return np.transpose(np.array(map(lambda x: math.sqrt(x[0]), matcopy.sum(axis=1).tolist()), ndmin=2))


def avg_row_norm(mat):
    row_norms = map(lambda x: x[0], row_wise_norm(mat).tolist())
    return sum(row_norms) / len(row_norms)


def non_zero_row_indexes(mat):
    return map(lambda x: x[0], filter(lambda x: x[1][0] != 0, enumerate(mat.sum(axis=1).tolist())))


def tf_idf_scores(mat, vocab):
    tfidf_trans = TfidfTransformer()
    tfidf_mat = tfidf_trans.fit_transform(mat)
    token_scores = list(enumerate(tfidf_mat.sum(axis=0).tolist()[0]))

    text_token_indexes = sorted(map(lambda i: i[1], filter(lambda i: i[0][:2] == "t:", vocab.items())))
    formula_token_indexes = sorted(map(lambda i: i[1], filter(lambda i: i[0][:2] != "t:", vocab.items())))

    if len(text_token_indexes) != 0:
        text_token_scores = sorted(itemgetter(*text_token_indexes)(token_scores), key=lambda x: x[1], reverse=True)
    else:
        text_token_scores = []

    if len(formula_token_indexes) != 0:
        formula_token_scores = sorted(itemgetter(*formula_token_indexes)(token_scores), key=lambda x: x[1], reverse=True)
    else:
        formula_token_scores = []

    return text_token_scores, formula_token_scores


def select_best_tokens(text_token_scores, formula_token_scores, config_args):
    if "intended_amount_of_text_tokens" in config_args:
        best_text_token_indexes = map(lambda x: x[0], text_token_scores[:config_args["intended_amount_of_text_tokens"]])
    else:
        best_text_token_indexes = map(lambda x: x[0], text_token_scores)

    if "intended_amount_of_formula_tokens" in config_args:
        best_formula_token_indexes = map(lambda x: x[0], formula_token_scores[:config_args["intended_amount_of_formula_tokens"]])
    else:
        best_formula_token_indexes = map(lambda x: x[0], formula_token_scores)

    return best_text_token_indexes, best_formula_token_indexes


def combine_with_equal_scale(mat1, mat2):
    if mat1.shape[1] == 0:
        return mat2
    if mat2.shape[1] == 0:
        return mat1

    float_mat1 = matrix_prune(mat1, 1.0)
    pruned_mat2 = matrix_prune(mat2, avg_row_norm(mat1) / avg_row_norm(mat2))

    return vertically_append_matrix(float_mat1, pruned_mat2)


if __name__ == "__main__":
    db = connect_to_db()
    cursor = db.cursor()

    interesting_settings = [
        {"data_basis": "only_theorems", "token_method": "lin", "granularity": "paragraphs"},
        {"data_basis": "only_theorems", "token_method": "lin", "granularity": "documents"},
        {"data_basis": "only_theorems", "token_method": "kristianto", "granularity": "paragraphs"},
        {"data_basis": "only_theorems", "token_method": "kristianto", "granularity": "documents"},
        {"data_basis": "only_theorems", "token_method": "plaintext", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "plaintext", "granularity": "documents"},
        # {"data_basis": "full_text", "token_method": "lin", "granularity": "documents"},
        # {"data_basis": "full_text", "token_method": "kristianto", "granularity": "documents"},
        # {"data_basis": "full_text", "token_method": "plaintext", "granularity": "documents"}
    ]

    for setting in interesting_settings:
        print "setting: " + str(setting)
        data_basis = setting["data_basis"]
        token_method = setting["token_method"]
        granularity = setting["granularity"]

        # === Config
        config_args = {
            "debug_max_items": None,
            "lin_max_token_length": 200
        }

        if token_method == "lin" or token_method == "kristianto":
            config_args["intended_amount_of_formula_tokens"] = 10000
            config_args["intended_amount_of_text_tokens"] = 10000
        elif token_method == "plaintext":
            config_args["intended_amount_of_formula_tokens"] = None
            config_args["intended_amount_of_text_tokens"] = 20000
        else:
            raise ValueError("token_method must be either 'lin', 'kristianto' or 'plaintext'")

        # === calc word counts
        # if granularity == "paragraphs":
        #     paragraph_generator = get_all_paragraphs_as_token_list(token_method, data_basis, cursor, config_args=config_args)
        #     token_counts = calc_word_counts(paragraph_generator)
        # elif granularity == "documents":
        #     document_generator = get_all_documents_as_token_list(token_method, data_basis, cursor, config_args=config_args)
        #     token_counts = calc_word_counts(document_generator)
        # else:
        #     raise ValueError("granularity must be either paragraphs or documents")

        # f = open("derived_data/" + setting_string(**setting) + "__token_counts.json", "w")
        # f.write(json.dumps(token_counts))
        # f.close()

        # === build text token dict
        # token_counts = json.load(open("derived_data/" + setting_string(**setting) + "__token_counts.json"))
        # text_token_dict = build_text_token_dict(token_counts, 3)

        # with open("derived_data/" + setting_string(**setting) + "__token2index_map.json", "w") as outfile:
        #     json.dump(text_token_dict, outfile)

        # === create raw csr_matrix for theorems
        # token2index_map = json.load(open("derived_data/" + setting_string(**setting) + "__token2index_map.json"))
        # if granularity == "paragraphs":
        #     paragraph_generator = get_all_paragraphs_as_token_list(token_method, data_basis, cursor, config_args=config_args)
        #     matrix, id_log = build_raw_csr_matrix(paragraph_generator, token2index_map)
        # elif granularity == "documents":
        #     document_generator = get_all_documents_as_token_list(token_method, data_basis, cursor, config_args=config_args)
        #     matrix, id_log = build_raw_csr_matrix(document_generator, token2index_map)
        # else:
        #     raise ValueError("granularity must be either paragraphs or documents")

        # save_csr_matrix(matrix, "derived_data/" + setting_string(**setting) + "__raw_tdm")

        # f = open("derived_data/" + setting_string(**setting) + "__ids", "w")
        # if granularity == "paragraphs":
        #     for id in id_log:
        #         f.write(id[0] + ";" + id[1] + "\n")
        # elif granularity == "documents":
        #     for id in id_log:
        #         f.write(id + "\n")
        # f.close()

        # === train and dump tf-idf model for theorem texts
        # raw_tdm = load_csr_matrix("derived_data/" + setting_string(**setting) + "__raw_tdm.npz")
        # tfidf_trans = TfidfTransformer()
        # tfidf_trans.fit(raw_tdm)

        # joblib.dump(tfidf_trans, "models/" + setting_string(**setting) + "__tfidf_model")

        # === retrieve best tf-idf terms
        # raw_tdm = load_csr_matrix("derived_data/" + setting_string(**setting) + "__raw_tdm.npz")
        # nz_row_indexes = non_zero_row_indexes(raw_tdm)

        # raw_tdm = raw_tdm[nz_row_indexes, :]
        # token2index_map = json.load(open("derived_data/" + setting_string(**setting) + "__token2index_map.json"))
        # index2token_map = {index: token for token, index in token2index_map.items()}

        # text_token_scores, formula_token_scores = tf_idf_scores(raw_tdm, token2index_map)
        # best_text_token_indexes, best_formula_token_indexes = select_best_tokens(text_token_scores, formula_token_scores, config_args)

        # processed_tdm = combine_with_equal_scale(raw_tdm[:, best_text_token_indexes], raw_tdm[:, best_formula_token_indexes])

        # new_index2old_index_map = {new_index: old_index for new_index, old_index in enumerate(best_text_token_indexes)}
        # new_index2old_index_map.update({new_index+len(best_text_token_indexes): old_index for new_index, old_index in enumerate(best_formula_token_indexes)})

        # new_token2index_map = {}
        # for new_index, old_index in new_index2old_index_map.items():
        #     new_token2index_map[index2token_map[old_index]] = new_index

        # f = open("derived_data/" + setting_string(**setting) + "__ids")
        # count = 0
        # new_ids = []
        # for line in f:
        #     if count in nz_row_indexes:
        #         new_ids.append(line.strip())

        #     count += 1
        # f.close()

        # # save processed tdm
        # save_csr_matrix(processed_tdm, "derived_data/" + setting_string(**setting) + "__processed_tdm")

        # # save respective ids
        # f = open("derived_data/" + setting_string(**setting) + "__processed_ids", "w")
        # for id in new_ids:
        #     f.write(id + "\n")
        # f.close()

        # # save token2index map
        # with open("derived_data/" + setting_string(**setting) + "__processed_token2index_map.json", "w") as outfile:
        #     json.dump(new_token2index_map, outfile)

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
        tdm = load_csr_matrix("derived_data/" + setting_string(**setting) + "__processed_tdm.npz")
        model = lda.LDA(n_topics=250)
        model.fit(tdm)
        joblib.dump(model, "models/" + setting_string(**setting) + "__topic_model")

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
        # frequent_msc_classes = ["05", "11", "14", "17", "20", "35", "37", "46", "53", "57", "60", "68", "81", "82"]
        # msc_classes = [
        #     "00", "01", "03", "05", "06", "08", "11", "12", "13", "14", "15", "16", "17", "18", "19",
        #     "20", "22", "26", "28", "30", "31", "32", "33", "34", "35", "37", "39", "40", "41", "42",
        #     "43", "44", "45", "46", "47", "49", "51", "52", "53", "54", "55", "57", "58", "60", "62",
        #     "65", "67", "68", "70", "74", "76", "78", "80", "81", "82", "83", "85", "86", "90", "91",
        #     "92", "93", "94", "97"
        # ]

        # # read doc2msc map
        # f = open("raw_data/doc2msc", "r")
        # doc2msc_map = {}
        # for line in f:
        #     x = line.split(";")
        #     doc2msc_map[x[0].strip()] = x[1].strip()
        # f.close()

        # # read document ids
        # f = open("derived_data/" + setting_string(**setting) + "__processed_ids")
        # doc_ids = []
        # for line in f:
        #     x = line.strip()
        #     doc_ids.append(x)

        # frequent_msc_doc_indexes = []
        # for index, doc_id in enumerate(doc_ids):
        #     if doc_id in doc2msc_map and doc2msc_map[doc_id][:2] in frequent_msc_classes:
        #         frequent_msc_doc_indexes.append(index)

        # raw_tdm = load_csr_matrix("derived_data/" + setting_string(**setting) + "__processed_tdm.npz")[frequent_msc_doc_indexes, :].astype(np.float)
        # doc_ids = itemgetter(*frequent_msc_doc_indexes)(doc_ids)

        # norm_tdm = Normalizer().fit_transform(raw_tdm)
        # tfidf_tdm = TfidfTransformer().fit_transform(raw_tdm)

        # mats = [(raw_tdm, "raw")]
        # for dim in [250, 300, 350]:
        #     for tfidf in [True]:
        #         for norm in [True]:
        #             label = "_".join(filter(lambda x: x is not None, [("tfidf" if tfidf else None), ("norm" if norm else None), ("lsi" + str(dim))]))
        #             print 'calc ' + label
        #             mat = raw_tdm
        #             if tfidf:
        #                 mat = TfidfTransformer().fit_transform(mat)
        #             if norm:
        #                 mat = normalize(mat)

        #             mat = TruncatedSVD(n_components=dim).fit_transform(raw_tdm)
        #             mats.append((mat, label))

        # measures = []
        # for target_class in frequent_msc_classes:
        #     for tdm, label in mats:
        #         # create label map
        #         label_vector = map(lambda id: None if id not in doc2msc_map else (1 if doc2msc_map[id][:2] == target_class else 0), doc_ids)

        #         num_positives = 0
        #         for x in label_vector:
        #             if x == 1:
        #                 num_positives += 1

        #         # determine train and test sets
        #         test_indexes = []
        #         train_indexes = []
        #         random.seed(0)

        #         count = 0
        #         for doc_id in doc_ids:
        #             if doc_id in doc2msc_map:
        #                 if random.random() > 0.2:
        #                     train_indexes.append(count)
        #                 else:
        #                     test_indexes.append(count)

        #             count += 1

        #         train_matrix = tdm[train_indexes, :]
        #         test_matrix = tdm[test_indexes, :]
        #         train_labels = itemgetter(*train_indexes)(label_vector)
        #         test_labels = itemgetter(*test_indexes)(label_vector)

        #         if num_positives > 0:
        #             # train classifier
        #             clf = svm.LinearSVC()
        #             clf.fit(train_matrix, train_labels)
        #             predictions = clf.predict(test_matrix).tolist()

        #             # evaluate
        #             evaluation_classes = map(lambda x: "tp" if x[0] == 1 and x[1] == 1 else ("fp" if x[0] == 1 and x[1] == 0 else ("fn" if x[0] == 0 and x[1] == 1 else "tn")), zip(predictions, test_labels))
        #             grouped_evaluation_classes = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        #             grouped_evaluation_classes.update(group_and_count(evaluation_classes))

        #             if grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fp"] == 0:
        #                 precision = None
        #             else:
        #                 precision = float(grouped_evaluation_classes["tp"])/(grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fp"])

        #             if grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fn"] == 0:
        #                 recall = None
        #             else:
        #                 recall = float(grouped_evaluation_classes["tp"])/(grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fn"])

        #             if precision is None or recall is None or precision+recall == 0.0:
        #                 f1 = None
        #             else:
        #                 f1 = 2*(precision*recall)/(precision+recall)
        #         else:
        #             precision = None
        #             recall = None
        #             f1 = None

        #         print label
        #         print "class: " + target_class
        #         print "num positives: " + str(num_positives)
        #         print "precision : " + str(precision)
        #         print "recall: " + str(recall)
        #         print "f1: " + str(f1)
        #         print ""

        #         measures.append({"target_class": target_class, "label": label, "num_positives": num_positives, "precision": precision, "recall": recall, "f1": f1})

        # with open("measures7.json", "w") as f:
        #     json.dump(measures, f)

        # with open("labels7.json", "w") as f:
        #     json.dump(map(lambda x: x[1], mats), f)

        # measures = json.load(open("measures6.json"))
        # labels = json.load(open("labels6.json"))
        # for cl in msc_classes:
        #     m = filter(lambda x: x['target_class'] == cl, measures)
        #     res = sorted(map(lambda x: (x['label'], x['f1']), filter(lambda x: x['f1'] is not None and x['num_positives'] > 250, m)), key=lambda x: x[1], reverse=True)
        #     if len(res) != 0:
        #         print "class: " + cl
        #         # print "num_positives: " + str(m[0]['num_positives'])
        #         print res
        #         # l1 = 'raw'
        #         # l2 = 'lsi750'
        #         # for l in [l1, l2]:
        #         #     if l not in res:
        #         #         res[l] = 0.0

        #         # print res[l1] - res[l2]
        #         # print "results: " + str(res)
        #         print ""

        # akku = {}
        # # for l in ['raw', 'tfidf', 'lsi250', 'lsi500', 'tfidflsi250', 'tfidtlsi500']: # labels for measures.json
        # # for l in ["raw", "norm", "lsi500", "lsi750"]: # labes for measures2.json
        # for l in labels:
        #     akku[l] = 0.0

        #     me = filter(lambda x: x['label'] == l and x['num_positives'] > 250, measures)
        #     for m in me:
        #         f1 = m['f1']
        #         if f1 is not None:
        #             akku[l] += f1

        # print list(sorted(akku.items(), key=lambda x: x[1], reverse=True))
