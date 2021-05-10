import glob
from collections import Counter

import nltk


def print_most_common_words_backup(n=100):
    vocab = []
    with open("barracuda_networks/aclImdb/imdb.vocab") as f:
        for word in f:
            word = word.strip('\n')
            vocab.append(word)
    word_index_to_count = {}
    with open("barracuda_networks/aclImdb/train/labeledBow.feat") as f:
        for line in f:
            word_to_counts = line.split(" ")[1:]
            for word_to_count in word_to_counts:
                word_index = int(word_to_count.split(":")[0])
                word_count = int(word_to_count.split(":")[1])
                if word_index not in word_index_to_count:
                    word_index_to_count[word_index] = word_count

                else:
                    word_index_to_count[word_index] = word_index_to_count[word_index] + word_count

    with open("barracuda_networks/aclImdb/test/labeledBow.feat") as f:
        for line in f:
            word_to_counts = line.split(" ")[1:]
            for word_to_count in word_to_counts:
                word_index = int(word_to_count.split(":")[0])
                word_count = int(word_to_count.split(":")[1])
                if word_index not in word_index_to_count:
                    word_index_to_count[word_index] = word_count

                else:
                    word_index_to_count[word_index] = word_index_to_count[word_index] + word_count

    temp = [{'index': k, 'count': v} for k, v in word_index_to_count.items()]
    sorted_word_index_to_count = sorted(temp, key=lambda x: x['count'], reverse=True)
    for word_index_to_count in sorted_word_index_to_count[0:n]:
        print(f"{vocab[word_index_to_count['index']]} - count: {word_index_to_count['count']}, index: {word_index_to_count['index']} ")


def print_most_common_words(n=110):
    reviews, _ = get_reviews_as_list_of_texts()
    tokenized_reviews = []
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    for review in reviews:
        tokenized_reviews.append(tokenizer.tokenize(review))

    flat_list = []
    for tokenized_review in tokenized_reviews:
        for word in tokenized_review:
            flat_list.append(word)

    counter = Counter(flat_list)
    for element in counter.most_common()[0:n]:
        print(f"{element[0]} - count: {element[1]}")


def get_reviews_as_list_of_texts(return_file_names=False) -> any:
    reviews = []
    target_line = None
    file_names = []
    # for filepath in glob.iglob('barracuda_networks/aclImdb/train/pos/*.txt'):
    #     # print(filepath)
    #     with open(filepath) as f:
    #         for line in f:
    #             reviews.append(line)
    #
    for filepath in glob.iglob('barracuda_networks/aclImdb/train/neg/*.txt'):
        with open(filepath) as f:
            for line in f:
                if "3316_2.txt" in filepath:
                    print(f"FOUND TARGET LINE OF: {line}")
                    target_line = line
                # reviews.append(line)
    #
    # for filepath in glob.iglob('barracuda_networks/aclImdb/test/pos/*.txt'):
    #     with open(filepath) as f:
    #         for line in f:
    #             reviews.append(line)
    #
    # for filepath in glob.iglob('barracuda_networks/aclImdb/test/neg/*.txt'):
    #     with open(filepath) as f:
    #         for line in f:
    #             reviews.append(line)

    for filepath in glob.iglob('barracuda_networks/aclImdb/train/unsup/*.txt'):
        # print(filepath)
        with open(filepath) as f:
            for line in f:
                reviews.append(line)
                file_names.append(filepath)

    print(len(reviews))
    if return_file_names:
        return reviews, target_line, file_names
    return reviews, target_line

if __name__ == "__main__":
    # print_most_common_words()

    get_reviews_as_list_of_texts(return_file_names=True)

