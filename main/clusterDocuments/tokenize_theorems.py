from util import flatten, connect_to_db, get_all_document_ids, add_to_dict, FormulaTokenizer, MixedTextSeparator
from util import TextTokenizer, group_and_count, build_csr_matrix, save_csr_matrix, load_csr_matrix
from util import vertically_append_matrix, non_zero_row_indexes, avg_row_norm, row_wise_norm, element_wise_multiply
from util import tokenize_mixed_text, get_paragraphs_from_doc, get_formulas_from_doc, get_docs_paragraphs_as_token_list
from util import get_all_docs_paragrahps_as_token_list, get_all_documents_as_feature_map, get_all_documents_as_token_list
from util import config, cursor
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
from numpy.linalg import norm
import itertools
import math
from fractions import Fraction
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, Normalizer
from pprint import pprint
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
import os.path

from glove import Glove, Corpus
# from gensim.models.ldamodel import LdaModel

msc_classes = [
    "00", "01", "03", "05", "06", "08", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "22", "26", "28", "30", "31", "32", "33", "34", "35", "37", "39", "40", "41", "42",
    "43", "44", "45", "46", "47", "49", "51", "52", "53", "54", "55", "57", "58", "60", "62",
    "65", "67", "68", "70", "74", "76", "78", "80", "81", "82", "83", "85", "86", "90", "91",
    "92", "93", "94", "97"
]

frequent_msc_classes = ["05", "11", "14", "17", "20", "35", "37", "46", "53", "57", "60", "68", "81", "82"]

removed_mathematicians = ["milet", "pythagoras", "knidos", "euklid", "archimedes", "apollonios",
                          "diophant", "heron", "aryabhata", "brahmagupta", "alhazen", "regiomontanus",
                          "stifel", "cardano", "wallis", "descartes", "lobatschewski", "tschebyschow",
                          "kowalewskaja", "kolmogorow", "wiles"]

mathematicians = ["fibonacci", "kepler", "fermat", "pascal", "bernoulli", "leibniz", "newton",
                  "euler", "lagrange", "monge", "laplace", "legendre", "fourier", "gauss",
                  "cauchy", "abel", "jacobi", "dirichlet", "galois", "weierstrass", "cayley", "hermite",
                  "kronecker", "riemann", "dedekind", "cantor", "klein", "hilbert",
                  "minkowski", "hausdorff", "lebesgue", "weyl", "ramanujan", "banach",
                  "neumann", "weil", "turing", "serre", "grothendieck", "perelman"]

force_gen = False if config("force_gen") is None else config("force_gen")


def calc_word_counts(items):
    global_token_counts = {}
    item_count = 1
    for id, tokens in items:
        print str(id) + " (" + str(item_count) + ")"
        add_to_dict(global_token_counts, group_and_count(tokens))
        item_count += 1

    return global_token_counts


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


def tf_idf_scores(mat, vocab):
    tfidf_trans = TfidfTransformer()
    tfidf_mat = tfidf_trans.fit_transform(mat)
    token_scores = list(enumerate(tfidf_mat.sum(axis=0).tolist()[0]))

    text_token_indexes = sorted(map(lambda i: i[1], filter(lambda i: i[0][:2] == "t:", vocab.items())))
    formula_token_indexes = sorted(map(lambda i: i[1], filter(lambda i: i[0][:2] != "t:", vocab.items())))

    if len(text_token_indexes) != 0:
        text_token_scores = itemgetter(*text_token_indexes)(token_scores)
    else:
        text_token_scores = []

    if len(formula_token_indexes) != 0:
        formula_token_scores = itemgetter(*formula_token_indexes)(token_scores)
    else:
        formula_token_scores = []

    return text_token_scores, formula_token_scores


def select_best_tokens(text_token_scores, formula_token_scores, intended_amount_of_text_tokens=None, intended_amount_of_formula_tokens=None):
    sorted_text_token_scores = sorted(text_token_scores, key=lambda x: x[1], reverse=True)
    sorted_formula_token_scores = sorted(formula_token_scores, key=lambda x: x[1], reverse=True)

    if intended_amount_of_text_tokens is not None:
        best_text_token_indexes = map(lambda x: x[0], sorted_text_token_scores[:intended_amount_of_text_tokens])
    else:
        best_text_token_indexes = map(lambda x: x[0], sorted_text_token_scores)

    if intended_amount_of_formula_tokens is not None:
        best_formula_token_indexes = map(lambda x: x[0], sorted_formula_token_scores[:intended_amount_of_formula_tokens])
    else:
        best_formula_token_indexes = map(lambda x: x[0], sorted_formula_token_scores)

    return best_text_token_indexes, best_formula_token_indexes


def matrix2document_token_weights(mat, ids, index2token_map):
    id_iter = iter(ids)
    document_maps = {}
    count = 0

    for row in mat:
        id = next(id_iter)
        non_zero_ind = row.nonzero()[1].tolist()
        non_zero_values = row[0, non_zero_ind].todense().tolist()[0]
        document_map = dict(zip(map(lambda x: index2token_map[x], non_zero_ind), non_zero_values))
        document_maps[id] = document_map

        if count % 1000 == 0:
            print count
        count += 1
    return document_maps


def get_author_msc_matrix():
    if not force_gen and all(os.path.isfile(filename) for filename in [
            "derived_data/author_msc_map.npz",
            "derived_data/author_msc_map__row2author_name.json",
            "derived_data/author_msc_map__col2msc_code.json"]):

        mat = load_csr_matrix("derived_data/author_msc_map.npz")
        with open("derived_data/author_msc_map__row2author_name.json") as f:
            row2author_map = json.load(f)

        with open("derived_data/author_msc_map__col2msc_code.json") as f:
            col2msc_map = json.load(f)

        return mat, row2author_map, col2msc_map
    else:
        author2msc_map = defaultdict(lambda: defaultdict(int))
        cursor().execute("""SELECT display_name, msc, COUNT(*) FROM authorship
                            JOIN msc_assignment ON authorship.document = msc_assignment.document
                        WHERE authorship.rank <= 2 AND msc_assignment.pos <= 3
                        GROUP BY display_name, msc
                        ORDER BY display_name""")

        for row in cursor():
            author2msc_map[row[0]][row[1][:2]] += row[2]

        author_names, msc_counts = zip(*author2msc_map.items())

        msc_code2index_map = dict(zip(msc_classes, range(len(msc_classes))))
        col2msc_map = {index: msc for msc, index in msc_code2index_map.items()}

        mat = build_csr_matrix(msc_counts, token2index_map=msc_code2index_map)
        save_csr_matrix(mat, "derived_data/author_msc_map")

        row2author_map = dict(zip(range(len(author_names)), author_names))
        with open("derived_data/author_msc_map__row2author_name.json", "w") as f:
            json.dump(row2author_map, f)

        with open("derived_data/author_msc_map__col2msc_code.json", "w") as f:
            json.dump(col2msc_map, f)

        return mat, row2author_map, col2msc_map


def author_query_by_msc(q_author, top_n=20, metric='cosine'):
    mat, row2author_map, col2msc_map = get_author_msc_matrix()
    x = [index for index, author in row2author_map.items() if author == q_author]
    if len(x) < 1:
        raise Exception("The author name '" + q_author + "' was not found")
    if len(x) > 1:
        raise Exception("The author name '" + q_author + "' is ambigue")

    author_index = x[0]

    distances = pairwise_distances(mat[author_index, :], Y=mat, metric=metric)
    top_n_indexes = list(itertools.islice(np.argsort(distances).tolist()[0], None, top_n))

    result_author_distances = distances[0, top_n_indexes].tolist()
    result_author_names = map(lambda index: row2author_map[str(index)], top_n_indexes)

    print zip(result_author_names, result_author_distances)


def get_token_counts(setting):
    if not force_gen and os.path.isfile("derived_data/" + setting_string(**setting) + "__token_counts.json"):
        with open("derived_data/" + setting_string(**setting) + "__token_counts.json") as f:
            return json.load(f)
    else:
        if setting['granularity'] == "paragraphs":
            paragraph_generator = get_all_docs_paragrahps_as_token_list(setting['token_method'], setting['data_basis'])
            token_counts = calc_word_counts(paragraph_generator)
        elif setting['granularity'] == "documents":
            document_generator = get_all_documents_as_feature_map(setting['token_method'], setting['data_basis'])
            token_counts = calc_word_counts(document_generator)
        else:
            raise ValueError("granularity must be either paragraphs or documents")

        with open("derived_data/" + setting_string(**setting) + "__token_counts.json", "w") as f:
            json.dump(token_counts, f)

        return token_counts


def get_token2index_map(setting):
    if not force_gen and os.path.isfile("derived_data/" + setting_string(**setting) + "__token2index_map.json"):
        with open("derived_data/" + setting_string(**setting) + "__token2index_map.json") as f:
            return json.load(f)
    else:
        token_counts = get_token_counts(setting)

        min_count = 1
        if config('min_token_count') is not None:
            min_count = config('min_token_count')

        frequent_tokens = map(lambda i: i[0], filter(lambda c: c[1] >= min_count, token_counts.items()))
        token2index_map = dict(zip(sorted(frequent_tokens), range(len(frequent_tokens))))

        with open("derived_data/" + setting_string(**setting) + "__token2index_map.json", "w") as f:
            json.dump(token2index_map, f)

        return token2index_map


def get_raw_tdm(setting):
    if not force_gen and all(os.path.isfile(filename) for filename in [
            "derived_data/" + setting_string(**setting) + "__raw_tdm.npz",
            "derived_data/" + setting_string(**setting) + "__ids"]):

        mat = load_csr_matrix("derived_data/" + setting_string(**setting) + "__raw_tdm.npz")
        ids = []
        with open("derived_data/" + setting_string(**setting) + "__ids") as f:
            count = 0
            for line in f:
                x = line.split(";")
                if setting['granularity'] == 'paragraphs':
                    ids.append((count, (x[0], x[1].strip())))
                elif setting['granularity'] == 'documents':
                    ids.append((count, x[0].strip()))
                else:
                    raise ValueError("granularity must be either 'documents' or 'paragraphs'")

        row2id_map = dict(ids)
        token2index_map = get_token2index_map(setting)
        column2token_map = {index: token for token, index in token2index_map.items()}

        return mat, row2id_map, column2token_map
    else:
        token2index_map = get_token2index_map(setting)
        column2token_map = {index: token for token, index in token2index_map.items()}

        if setting['granularity'] == "paragraphs":
            paragraph_generator = get_all_docs_paragrahps_as_token_list(setting['token_method'], setting['data_basis'])
            mat, id_log = build_raw_csr_matrix(paragraph_generator, token2index_map)
        elif setting['granularity'] == "documents":
            document_generator = get_all_documents_as_feature_map(setting['token_method'], setting['data_basis'])
            mat, id_log = build_raw_csr_matrix(document_generator, token2index_map)
        else:
            raise ValueError("granularity must be either paragraphs or documents")

        save_csr_matrix(mat, "derived_data/" + setting_string(**setting) + "__raw_tdm")

        f = open("derived_data/" + setting_string(**setting) + "__ids", "w")
        if setting['granularity'] == "paragraphs":
            for id in id_log:
                f.write(id[0] + ";" + id[1] + "\n")
        elif setting['granularity'] == "documents":
            for id in id_log:
                f.write(id + "\n")
        f.close()

        row2id_map = dict(zip(range(len(id_log)), id_log))

        return mat, row2id_map, column2token_map


def get_processed_tdm(setting, intended_amount_of_text_tokens=None, intended_amount_of_formula_tokens=None):
    if not force_gen and all(os.path.isfile(filename) for filename in [
            "derived_data/" + setting_string(**setting) + "__processed_tdm.npz",
            "derived_data/" + setting_string(**setting) + "__processed_ids",
            "derived_data/" + setting_string(**setting) + "__processed_token2index_map.json"]):

        mat = load_csr_matrix("derived_data/" + setting_string(**setting) + "__processed_tdm.npz")
        ids = []
        with open("derived_data/" + setting_string(**setting) + "__processed_ids") as f:
            count = 0
            for line in f:
                x = line.split(";")
                if setting['granularity'] == 'paragraphs':
                    ids.append((count, (x[0], x[1].strip())))
                elif setting['granularity'] == 'documents':
                    ids.append((count, x[0].strip()))
                else:
                    raise ValueError("granularity must be either 'documents' or 'paragraphs'")

        row2id_map = dict(ids)
        with open("derived_data/" + setting_string(**setting) + "__processed_token2index_map.json") as f:
            token2index_map = json.load(f)

        column2token_map = {index: token for token, index in token2index_map.items()}

        return mat, row2id_map, column2token_map

    else:
        # retrieve best tf-idf terms
        raw_tdm, row2id_map, column2token_map = get_raw_tdm(setting)
        nz_row_indexes = non_zero_row_indexes(raw_tdm)
        raw_tdm = raw_tdm[nz_row_indexes, :]

        token2index_map = get_token2index_map(setting)

        text_token_scores, formula_token_scores = tf_idf_scores(raw_tdm, token2index_map)
        best_text_token_indexes, best_formula_token_indexes = select_best_tokens(text_token_scores, formula_token_scores, intended_amount_of_text_tokens, intended_amount_of_formula_tokens)

        text_tdm = raw_tdm[:, best_text_token_indexes]
        formula_tdm = raw_tdm[:, best_formula_token_indexes]
        if text_tdm.shape[1] == 0:
            processed_tdm = formula_tdm
        elif formula_tdm == 0:
            processed_tdm = text_tdm
        else:
            float_text_tdm = element_wise_multiply(text_tdm, 1.0)
            pruned_formula_tdm = element_wise_multiply(formula_tdm, avg_row_norm(text_tdm) / avg_row_norm(formula_tdm))
            processed_tdm = vertically_append_matrix(float_text_tdm, pruned_formula_tdm)

        new_index2old_index_map = {new_index: old_index for new_index, old_index in enumerate(best_text_token_indexes)}
        new_index2old_index_map.update({new_index+len(best_text_token_indexes): old_index for new_index, old_index in enumerate(best_formula_token_indexes)})

        new_token2index_map = {}
        for new_index, old_index in new_index2old_index_map.items():
            new_token2index_map[column2token_map[old_index]] = new_index

        new_column2token_map = {index: token for token, index in new_token2index_map.items()}

        new_row2id_map = {}
        count = 0
        for index, id in row2id_map.items():
            if index in nz_row_indexes:
                new_row2id_map[count] = id
                count += 1

        # save processed tdm
        save_csr_matrix(processed_tdm, "derived_data/" + setting_string(**setting) + "__processed_tdm")

        # save respective ids
        with open("derived_data/" + setting_string(**setting) + "__processed_ids", "w") as f:
            for index, id in sorted(new_row2id_map.items(), key=lambda x: x[0]):
                f.write(id + "\n")

        # save token2index map
        with open("derived_data/" + setting_string(**setting) + "__processed_token2index_map.json", "w") as outfile:
            json.dump(new_token2index_map, outfile)

        return processed_tdm, new_row2id_map, new_column2token_map


def get_glove_corpus_model(setting):
    if not force_gen and os.path.isfile("models/" + setting_string(**setting) + "__glove_corpus_model"):
        return Corpus.load("models/" + setting_string(**setting) + "__glove_corpus_model")
    else:
        token2index_map = json.load(open("derived_data/" + setting_string(**setting) + "__processed_token2index_map.json"))

        if setting['granularity'] == 'documents':
            item_generator = get_all_documents_as_token_list(setting['token_method'], setting['data_basis'])
        elif setting['granularity'] == 'paragraphs':
            item_generator = get_all_docs_paragrahps_as_token_list(setting['token_method'], setting['data_basis'])
        else:
            raise

        corpus = (filter(lambda token: token in token2index_map, doc[1]) for doc in item_generator)
        corpus_model = Corpus(dictionary=token2index_map)
        corpus_model.fit(corpus)
        corpus_model.save("models/" + setting_string(**setting) + "__glove_corpus_model")

        return corpus_model


def get_glove_model(setting, glove_n_components=300):
    if not force_gen and os.path.isfile("models/" + setting_string(**setting) + "__glove_model"):
        return Glove.load("models/" + setting_string(**setting) + "__glove_model")
    else:
        corpus_model = get_glove_corpus_model(setting)
        token2index_map = json.load(open("derived_data/" + setting_string(**setting) + "__processed_token2index_map.json"))

        glove_model = Glove(no_components=glove_n_components)
        glove_model.fit(corpus_model.matrix, no_threads=4)
        glove_model.add_dictionary(token2index_map)
        glove_model.save("models/" + setting_string(**setting) + "__glove_model")

        return glove_model


def get_lda_model(setting, lda_n_topics=250):
    if not force_gen and os.path.isfile("models/" + setting_string(**setting) + "__topic_model"):
        return joblib.load("models/" + setting_string(**setting) + "__topic_model")
    else:
        tdm, row2id_map, col2token_map = get_processed_tdm(setting)

        model = lda.LDA(n_topics=lda_n_topics)
        model.fit(tdm)
        joblib.dump(model, "models/" + setting_string(**setting) + "__topic_model")

        return model


def print_lda_topic_words(model, col2token_map, n_words):
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        print "Topic " + str(i) + ": " + repr(map(lambda x: col2token_map[x], np.argsort(topic_dist).tolist()[-n_words:]))
        print ""


def enrich_ids_with_authors(setting):
    f = open("derived_data/" + setting_string(**setting) + "__ids")
    f2 = open("derived_data/" + setting_string(**setting) + "__ids_with_authors", "w")

    findAuthorStmt = "SELECT display_name, zbmath_id FROM authorship WHERE document=%(document)s AND rank<=%(maxrank)s"
    max_author_rank = 2
    if config("max_author_rank") is not None:
        max_author_rank = config("max_author_rank")

    for line in f:
        x = line.split(";")
        ids = {}
        if setting['granularity'] == 'paragraphs':
            ids['doc'] = x[0].strip()
            ids['par'] = x[1].strip()
        elif setting['granularity'] == 'documents':
            ids['doc'] = x[0].strip()
        else:
            raise ValueError("granularity must be either 'paragraphs' or 'documents'")

        authors = []
        cursor().execute(findAuthorStmt, {"document": ids['doc'], "maxrank": max_author_rank})
        for row in cursor():
            display_name = row[0]
            zbmath_id = row[1]
            authors.append((display_name, zbmath_id))

        if setting['granularity'] == 'paragraphs':
            newline = ids['doc'] + ";" + ids['par'] + ";" + ";".join(map(lambda author: author[0] + "(" + str(author[1]) + ")", authors)) + (";" * (max_author_rank-len(authors)))
        elif setting['granularity'] == 'documents':
            newline = ids['doc'] + ";" + ";".join(map(lambda author: author[0] + "(" + str(author[1]) + ")", authors)) + (";" * (max_author_rank-len(authors)))
        else:
            raise ValueError("granularity must be either 'paragraphs' or 'documents'")

        f2.write(newline + "\n")

    f.close()
    f2.close()


def theorem_author_mappings(setting, ids_with_authors_file):
    author_theorem_map = {}
    theorem_author_map = {}
    index = 0

    f = open(ids_with_authors_file)
    for line in f:
        x = line.split(";")

        if setting['granularity'] == 'documents':
            id = x[0]
            offset = 1
        elif setting['granularity'] == 'paragraphs':
            id = (x[0], x[1])
            offset = 2

        authors = []
        for i in range(offset, len(x)):
            authors.append(x[i].strip())

        for author in authors:
            if author not in author_theorem_map:
                author_theorem_map[author] = []

            if index not in theorem_author_map:
                theorem_author_map[index] = []

            author_theorem_map[author].append((id, index))
            theorem_author_map[index].append(author)
        index += 1
    f.close()

    return theorem_author_map, author_theorem_map


def get_author_theorem_matrix(setting):
    if not force_gen and all(os.path.isfile(filename) for filename in [
            "derived_data/" + setting_string(**setting) + "__raw_author_matrix.npz",
            "derived_data/" + setting_string(**setting) + "__raw_author_matrix_row2author.json",
            "derived_data/" + setting_string(**setting) + "__raw_author_matrix_col2item.json"]):

        mat = load_csr_matrix("derived_data/" + setting_string(**setting) + "__raw_author_matrix.npz")
        with open("derived_data/" + setting_string(**setting) + "__raw_author_matrix_row2author.json") as f:
            row2author_map = json.load(f)

        with open("derived_data/" + setting_string(**setting) + "__raw_author_matrix_col2item.json") as f:
            col2item_map = json.load(f)

        return mat, row2author_map, col2item_map
    else:
        author_set = set()
        item_id_set = set()
        with open("derived_data/" + setting_string(**setting) + "__ids_with_authors") as f:
            for line in f:
                x = line.split(";")
                if setting['granularity'] == 'documents':
                    item_id_set.add(x[0])
                    offset = 1
                elif setting['granularity'] == 'paragraphs':
                    item_id_set.add((x[0], x[1]))
                    offset = 2
                else:
                    raise

                for i in range(offset, len(x)):
                    author_set.add(x[i].strip())

        count = 0
        item2index_map = {}
        with open("derived_data/" + setting_string(**setting) + "__processed_ids") as f:
            for line in f:
                item2index_map[line.strip()] = count
                count += 1

        # item2index_map = dict(zip(sorted(list(item_id_set)), range(len(item_id_set))))
        author2index_map = dict(zip(sorted(list(author_set)), range(len(author_set))))

        author_item_indexes = map(lambda x: {}, range(len(author2index_map)))
        with open("derived_data/" + setting_string(**setting) + "__ids_with_authors") as f:
            for line in f:
                x = line.split(";")
                if setting['granularity'] == 'documents':
                    item_index = item2index_map.get(x[0])
                    offset = 1
                elif setting['granularity'] == 'paragraphs':
                    item_index = item2index_map.get((x[0], x[1]))
                    offset = 2
                else:
                    raise

                if item_index is not None:
                    for i in range(offset, len(x)):
                        author_index = author2index_map[x[i].strip()]
                        author_item_indexes[author_index][item_index] = 1.0

        mat = build_csr_matrix(list_of_dicts=author_item_indexes, num_attributes=len(item2index_map))
        save_csr_matrix(mat, "derived_data/" + setting_string(**setting) + "__raw_author_matrix")

        row2author_map = {index: author for author, index in author2index_map.items()}
        with open("derived_data/" + setting_string(**setting) + "__raw_author_matrix_row2author.json", "w") as f:
            json.dump(row2author_map, f)

        col2item_map = {index: item for item, index in item2index_map.items()}
        with open("derived_data/" + setting_string(**setting) + "__raw_author_matrix_col2item.json", "w") as f:
            json.dump(col2item_map, f)

        return mat, row2author_map, col2item_map


def get_normed_author_theorem_matrix(setting):
    if (not force_gen
            and os.path.isfile("derived_data/" + setting_string(**setting) + "__normed_author_theorem_matrix.npz")
            and os.path.isfile("derived_data/" + setting_string(**setting) + "__normed_theorem_author_matrix.npz")):
        normed_author_theorem_mat = load_csr_matrix("derived_data/" + setting_string(**setting) + "__normed_author_theorem_matrix.npz")
        normed_theorem_author_mat = load_csr_matrix("derived_data/" + setting_string(**setting) + "__normed_theorem_author_matrix.npz")

        return normed_author_theorem_mat, normed_theorem_author_mat
    else:
        mat, r, c = get_author_theorem_matrix(setting)
        normed_author_theorem_mat = normalize(mat)
        normed_theorem_author_mat = normalize(mat.transpose())
        save_csr_matrix(normed_author_theorem_mat, "derived_data/" + setting_string(**setting) + "__normed_author_theorem_matrix")
        save_csr_matrix(normed_theorem_author_mat, "derived_data/" + setting_string(**setting) + "__normed_theorem_author_matrix")
        return normed_author_theorem_mat, normed_theorem_author_mat


def get_normed_theorem_topic_matrix(setting):
    if (not force_gen
            and os.path.isfile("derived_data/" + setting_string(**setting) + "__normed_theorem_topic_matrix.npy")
            and os.path.isfile("derived_data/" + setting_string(**setting) + "__normed_topic_theorem_matrix.npy")):
        normed_theorem_topic_mat = np.load("derived_data/" + setting_string(**setting) + "__normed_theorem_topic_matrix.npy")
        normed_topic_theorem_mat = np.load("derived_data/" + setting_string(**setting) + "__normed_topic_theorem_matrix.npy")

        return normed_theorem_topic_mat, normed_topic_theorem_mat
    else:
        model = get_lda_model(setting)
        theorem_topic_matrix = model.doc_topic_
        normed_theorem_topic_matrix = normalize(theorem_topic_matrix)
        normed_topic_theorem_matrix = normalize(theorem_topic_matrix.transpose())
        np.save("derived_data/" + setting_string(**setting) + "__normed_theorem_topic_matrix", normed_theorem_topic_matrix)
        np.save("derived_data/" + setting_string(**setting) + "__normed_topic_theorem_matrix", normed_topic_theorem_matrix)
        return normed_theorem_topic_matrix, normed_topic_theorem_matrix


def query_author_by_topics(setting, q_author):
    normed_theorem_topic_matrix, normed_topic_theorem_matrix = get_normed_theorem_topic_matrix(setting)
    normed_author_theorem_matrix, normed_theorem_author_matrix = get_normed_author_theorem_matrix(setting)

    mat, row2author, col2theorem = get_author_theorem_matrix(setting)
    author2row = {author: index for index, author in row2author.items()}

    a_th = normed_author_theorem_matrix[76123, :]
    p_to = a_th * normed_theorem_topic_matrix
    a_th = None

    p_th = np.dot(p_to, normed_topic_theorem_matrix)
    p_to = None

    res = p_th * normed_theorem_author_matrix
    return res


def query_author():
    theorem_author_map, author_theorem_map = theorem_author_mappings("derived_data/" + setting_string(**setting) + "__processed_ids_with_authors")
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
        theorem_vector = theorem_topic_matrix[index, :].tolist()

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
        topic_vector = theorem_topic_matrix[:, topic_index].tolist()
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


def classification_quality(tdm, label_vector):
    num_positives = 0
    for x in label_vector:
        if x == 1:
            num_positives += 1

    # determine train and test sets
    test_indexes = []
    train_indexes = []
    random.seed(0)

    for i in range(tdm.shape[0]):
        if random.random() > 0.2:
            train_indexes.append(i)
        else:
            test_indexes.append(i)

    train_matrix = tdm[train_indexes, :]
    test_matrix = tdm[test_indexes, :]
    train_labels = itemgetter(*train_indexes)(label_vector)
    test_labels = itemgetter(*test_indexes)(label_vector)

    if num_positives > 0:
        # train classifier
        clf = svm.LinearSVC()
        clf.fit(train_matrix, train_labels)
        predictions = clf.predict(test_matrix).tolist()

        # evaluate
        evaluation_classes = map(lambda x: "tp" if x[0] == 1 and x[1] == 1 else ("fp" if x[0] == 1 and x[1] == 0 else ("fn" if x[0] == 0 and x[1] == 1 else "tn")), zip(predictions, test_labels))
        grouped_evaluation_classes = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        grouped_evaluation_classes.update(group_and_count(evaluation_classes))

        if grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fp"] == 0:
            precision = None
        else:
            precision = float(grouped_evaluation_classes["tp"])/(grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fp"])

        if grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fn"] == 0:
            recall = None
        else:
            recall = float(grouped_evaluation_classes["tp"])/(grouped_evaluation_classes["tp"] + grouped_evaluation_classes["fn"])

        if precision is None or recall is None or precision+recall == 0.0:
            f1 = None
        else:
            f1 = 2*(precision*recall)/(precision+recall)
    else:
        precision = None
        recall = None
        f1 = None

    print f1

if __name__ == "__main__":
    # === mix kristianto and lin features
    """kristianto_tdm = load_csr_matrix("derived_data/full_text_kristianto_documents__processed_tdm.npz")
    kristianto_ids = []
    with open("derived_data/full_text_kristianto_documents__processed_ids") as f:
        for line in f:
            x = line.strip()
            kristianto_ids.append(x)

    kristianto_token2index_map = json.load(open("derived_data/full_text_kristianto_documents__processed_token2index_map.json"))
    kristianto_index2token_map = {index: token for token, index in kristianto_token2index_map.items()}
    kristianto_document_token_weights = matrix2document_token_weights(kristianto_tdm, kristianto_ids, kristianto_index2token_map)

    lin_tdm = load_csr_matrix("derived_data/full_text_lin_documents__processed_tdm.npz")
    lin_ids = []
    with open("derived_data/full_text_lin_documents__processed_ids") as f:
        for line in f:
            x = line.strip()
            lin_ids.append(x)

    lin_token2index_map = json.load(open("derived_data/full_text_lin_documents__processed_token2index_map.json"))
    lin_index2token_map = {index: token for token, index in lin_token2index_map.items()}
    lin_document_token_weights = matrix2document_token_weights(lin_tdm, lin_ids, lin_index2token_map)

    all_doc_ids = set(kristianto_document_token_weights.keys()).union(lin_document_token_weights.keys())
    unified_doc_maps = []
    for doc_id in all_doc_ids:
        lin_doc = lin_document_token_weights[doc_id]
        kristianto_doc = kristianto_document_token_weights[doc_id]

        unified_doc = lin_doc
        for token, value in kristianto_doc.items():
            if token in unified_doc and unified_doc[token] != kristianto_doc[token]:
                raise ValueError("different token values for token " + token + " in document " + doc_id + "(" + unified_doc[token] + " and " + kristianto_doc[token] + ")")
        unified_doc.update(kristianto_doc)

        unified_doc_maps.append(unified_doc)

    with open("foo", "w") as f:
        json.dump(unified_doc_maps, f)"""


if __name__ == "__main__":
    interesting_settings = [
        # {"data_basis": "only_theorems", "token_method": "lin", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "lin", "granularity": "documents"},
        # {"data_basis": "only_theorems", "token_method": "kristianto", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "kristianto", "granularity": "documents"},
        # {"data_basis": "only_theorems", "token_method": "plaintext", "granularity": "paragraphs"},
        # {"data_basis": "only_theorems", "token_method": "plaintext", "granularity": "documents"},
        {"data_basis": "full_text", "token_method": "lin", "granularity": "documents"},
        # {"data_basis": "full_text", "token_method": "kristianto", "granularity": "documents"},
        # {"data_basis": "full_text", "token_method": "plaintext", "granularity": "documents"}
    ]

    # author_query_by_msc("Shanjian Tang")

    for setting in interesting_settings:
        print "setting: " + str(setting)

        tuning_params = {
            "glove_no_components": 300,
            "lda_n_topics": 250
        }

        if setting['token_method'] == "lin" or setting['token_method'] == "kristianto":
            tuning_params["intended_amount_of_formula_tokens"] = 10000
            tuning_params["intended_amount_of_text_tokens"] = 10000
        elif setting['token_method'] == "plaintext":
            tuning_params["intended_amount_of_formula_tokens"] = None
            tuning_params["intended_amount_of_text_tokens"] = 20000
        else:
            raise ValueError("token_method must be either 'lin', 'kristianto' or 'plaintext'")

        print list(itertools.islice(reversed(np.argsort(query_author_by_topics(setting, "Bassam Kojok(None)")).tolist()[0]), None, 20))
        # # play with glove
        # glove_model = get_glove_model(setting)
        # token2index_map = json.load(open("derived_data/" + setting_string(**setting) + "__processed_token2index_map.json"))

        # word_vector = lambda token: glove_model.word_vectors[token2index_map[token]]
        # neighborhood = lambda vec: glove_model._similarity_query(vec, 20)
        # token_sequence2vector = lambda par: glove_model.transform_paragraph(filter(lambda x: x in token2index_map, par))

        # if setting['granularity'] == 'documents':
        #     generator = get_all_documents_as_token_list(setting['token_method'], setting['data_basis'])
        # elif setting['granularity'] == 'paragraphs':
        #     generator = get_all_docs_paragrahps_as_token_list(setting['token_method'], setting['data_basis'])
        # else:
        #     raise

        # # mathematicians_in_corpus = filter(lambda x: x in token2index_map, mathematicians)

        # count = 0
        # item_vectors = []
        # ids = []
        # for item in generator:
        #     if len(item[0]) != 0:
        #         ids.append(item[0])
        #         item_vectors.append(token_sequence2vector(item[1]))

        #     if count % 1000 == 0:
        #         print count
        #     count += 1

        # with open("derived_data/" + setting_string(**setting) + "__glove_tdm_ids", "w") as f:
        #     for id in ids:
        #         if setting['granularity'] == 'paragraphs':
        #             f.write(id[0] + ";" + id[1] + "\n")
        #         elif setting['granularity'] == 'documents':
        #             f.write(id + "\n")
        #         else:
        #             raise

        # np.save("derived_data/" + setting_string(**setting) + "__glove_tdm", np.array(item_vectors))

        # pprint(neighborhood(word_vector("t:trigonometric")))
        # pprint(glove_model._similarity_query(word_vector, 50))

        # transform matrix
        """raw_theorem_tdm = load_csr_matrix("derived_data/theorem_raw_tdm.npz")
        model = joblib.load("models/topic_model")
        for i in range(10):
            print model.doc_topic_[i, :].tolist()
            print ""
        """

        # === train svm

        # read doc2msc map
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

        # for target_class in frequent_msc_classes:
        #     label_vector = map(lambda id: None if id not in doc2msc_map else (1 if doc2msc_map[id][:2] == target_class else 0), doc_ids)
        #     classification_quality(raw_tdm, label_vector)
