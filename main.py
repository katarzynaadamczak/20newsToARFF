from sklearn.datasets import fetch_20newsgroups
import string
from itertools import islice
import arff

chosen_categories = ["rec.autos", "sci.med", "talk.politics.guns", "comp.graphics", "alt.atheism"]

prepositions = ["on", "in", "at", "since", "for", "ago", "before", "to", "past", "by", "under", "over", "above", "into", "from", "about", "an",
                "but", "of"]


def main():
    newsgroups_train = fetch_20newsgroups(subset="train", remove=('headers', 'footers', 'quotes'),
                                          categories=chosen_categories)
    # print(newsgroups_train.data[1])
    vectors = data_to_vectors(newsgroups_train)
    # for v in vectors[:5]:
    #     print(v)

    most_frequent_words = get_most_frequent_words(vectors)
    print(len(most_frequent_words))
    documents_vectors = make_vectors_for_arff(vectors, most_frequent_words)
    # arff.dump("words_results.arff", documents_vectors, relation="words", names=most_frequent_words)



def make_vectors_for_arff(vectors, most_frequent_words):
    """

    :param vectors: list of all documents where each document is [file id, categorie, {word:number_of_occurence}]
    :param most_frequent_words: list of 10k most frequent words in all texts
    :return: returns list of lists of numbers, where each number represents occurence of word from most_frequent_words in vectors
    """
    all_documents = []
    for document in vectors:
        single_document = []
        document_dict = document[2]
        for word in most_frequent_words:
            if word in document_dict:
                # print(word, document_dict[word])
                single_document.append(document_dict[word])
            else:
                single_document.append(0)
        # print("-------------------next--------------------")
        single_document.append(document[1])
        all_documents.append(single_document)
    return all_documents


def get_most_frequent_words(vectors):
    """

    :param vectors: list of vectors for each text file, which consists of [file id, categorie, {word:number_of_occurence}]
    :return: list of k most frequent words occuring in all texts
    """
    all_words = dict()
    for doc_dict in vectors:
        for key, value in doc_dict[2].items():
            all_words[key] = all_words[key] + value if key in all_words else value
    sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
    first_k_words = sorted_words[:10000]
    first_k_words_list = []
    for i in first_k_words:
        first_k_words_list.append(i[0])
        # first_k_words_list.append(i[1])
    return first_k_words_list




def data_to_vectors(bunch):
    """

    :param bunch: list of unformatted texts
    :return: list of vectors where each vector is [doc_id, category, word1, frequency_of _w1, …, wordn, frequency_of _wn]
    """
    list_of_vectors = []
    for i, text in enumerate(bunch.data):
        single_vector = []
        doc_id = get_doc_id_from_string(bunch.filenames[i])
        single_vector.append(doc_id)
        single_vector.append(bunch.target_names[bunch.target[i]])
        single_vector.append(turn_text_to_vector(text))
        list_of_vectors.append(single_vector)
    return list_of_vectors


def turn_text_to_vector(text) -> dict:
    """

    :param text: unformatted string
    :return: dictionary of words with number of their occurences
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    word_dict = dict()
    for word in text.split():
        word_fixed = word.lower()
        if word_fixed.isalpha() and len(word_fixed) > 1 and word_fixed not in prepositions:
            # print(word_fixed)
            word_dict[word_fixed] = word_dict[word_fixed] + 1 if word_fixed in word_dict else 1
    # words_vector = []
    # for key, value in word_dict.items():
    #     words_vector.extend([key, value])
    return word_dict


def get_doc_id_from_string(path):
    """

    :param path: file path
    :return: filename
    """
    return path.split("\\")[-1]


main()

