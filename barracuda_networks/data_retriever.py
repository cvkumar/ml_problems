import pandas as pd


def print_most_common_words(n=100):
    vocab = []
    with open("aclImdb/imdb.vocab") as f:
        for word in f:
            word = word.strip('\n')
            vocab.append(word)
    word_index_to_count = {}
    with open("aclImdb/train/labeledBow.feat") as f:
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
        print(f"index: {word_index_to_count['index']}, count: {word_index_to_count['count']}, 'word': {vocab[word_index_to_count['index']]}")


if __name__ == "__main__":
    print_most_common_words()
