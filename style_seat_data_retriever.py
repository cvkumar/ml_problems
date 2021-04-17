import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf

from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

RANDOM_STATE = 2020

CATEGORIES = ['Special Occasion Makeup', 'Twists', 'Special Occasion Style', 'Relaxers', 'Massage', 'Knots', 'Locs', 'Curling Treatments', 'Braids', 'Manicure', "Women's Haircut", 'Natural Hair', 'Hair Color', 'Weaves', 'Eyelashes', "Men's Color", 'Sugaring', 'Chemical Perms', 'Pedicure', 'Extensions', 'Eyebrows', 'Makeup', 'Sets & Styles', 'Style', 'Barber', "Men's Haircut", 'Wellness', 'Hair Treatments', 'Skincare', 'Waxing', 'Straightening Treatments', 'Highlights']

def create_df_for_category(dataset: pd.DataFrame, category: str):
    results = {
        'service_name': [],
        'service_category': []
    }

    print(f"creating dataset for category: {category}")

    for index, row in dataset.iterrows():
        service_name = row['service_name']
        entries = row['service_category'].split(',')
        entries = [entry.strip() for entry in entries]

        if category in entries:
            label = 1
        else:
            label = 0

        results['service_name'].append(service_name)
        results['service_category'].append(label)

    category_df = pd.DataFrame(data={'service_name': results['service_name'], 'service_category': results['service_category']})
    print(category_df['service_category'].value_counts())
    df_groupby = category_df.groupby("service_category")
    category_df = df_groupby.apply(
        lambda x: x.sample(df_groupby.size().min()).reset_index(drop=True)
    )
    print(category_df['service_category'].value_counts())
    return category_df

def train_model_for_category(category_df, category, test_size=.2):
    vocab_size = 10000
    embedding_dim = 16
    max_length = 50
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    using_test_set = test_size > 0
    if not using_test_set:
        train_df = category_df
        test_df = pd.DataFrame(data={'service_name': [], 'service_category': []})
    else:
        train_df, test_df = train_test_split(category_df, test_size=test_size)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    train_service_names = list(train_df['service_name'])
    train_labels = list(train_df['service_category'])
    test_service_names = list(test_df['service_name'])
    test_labels = list(test_df['service_category'])

    tokenizer.fit_on_texts(train_service_names)

    training_sequences = tokenizer.texts_to_sequences(train_service_names)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(test_service_names)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(train_labels)

    testing_padded = np.array(testing_padded)
    testing_labels = np.array(test_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 10
    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=0)

    if using_test_set:
        test_predictions_raw = model.predict(testing_padded)
        test_predictions = [1 if prediction[0] > .5 else 0 for prediction in test_predictions_raw]

        print(f"scores for category: {category}")
        print(metrics.classification_report(y_true=testing_labels, y_pred=test_predictions, digits=3))

    return model, tokenizer


if __name__ == "__main__":

    vocab_size = 10000
    embedding_dim = 16
    max_length = 50
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    df = pd.read_csv("data/styleseat/training_data.csv")
    train_df, test_df = train_test_split(df, test_size=.2)

    category_to_model = {}
    category_to_tokenizer = {}

    test_service_names = list(test_df['service_name'])
    test_service_categories = list(test_df['service_category'])

    final_predictions = [[] for x in test_service_names]

    categories = CATEGORIES

    for category in categories:
        category_df = create_df_for_category(dataset=train_df, category=category)
        model, tokenizer = train_model_for_category(category_df, category, test_size=0)
        category_to_model[category] = model
        category_to_tokenizer[category] = tokenizer

        testing_sequences = tokenizer.texts_to_sequences(test_service_names)
        testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        category_predictions_raw = model.predict(testing_padded)
        category_predictions = [1 if prediction[0] > .5 else 0 for prediction in category_predictions_raw]

        for i in range(len(category_predictions)):
            prediction = category_predictions[i]
            if prediction == 1:
                final_predictions[i].append(category)

    for i in range(len(final_predictions)):
        final_predictions[i] = ', '.join(final_predictions[i])

    correct = 0
    incorrect = 0

    for i in range(len(final_predictions)):
        prediction = str(final_predictions[i])
        truth_category = test_service_categories[i]

        prediction_set = set([entry.strip() for entry in prediction.split(',')])
        truth_category_set = set([entry.strip() for entry in truth_category.split(',')])

        if prediction_set == truth_category_set:
            correct += 1
        else:
            incorrect += 1


    print(f"correct: {correct}, incorrect: {incorrect}, accuracy: {correct / (correct+incorrect)}")

    #
    # train_predictions = []
    # for service_name in train_service_names:
    #     train_predictions.append(predict(model, tokenizer, service_name))

    # category_set = set()
    # categories = []
    # for index, row in df.iterrows():
    #     entries = row['service_category'].split(',')
    #     entries = [entry.strip() for entry in entries]
    #     for entry in entries:
    #         category_set.add(entry)
    #         categories.append(entry)
    #
    # counter = Counter(categories)
    # print(counter)
    # print(len(counter))
    #
    # print(list(category_set))
    #
    #
    # categories = []
    # for index, row in df.iterrows():
    #     entries = row['service_category'].split(',')
    #     entries = [entry.strip() for entry in entries]
    #     for entry in entries:
    #         categories.append(entry)
    #
    # counter = Counter(categories)
    # print(counter)
    # print(len(counter))
