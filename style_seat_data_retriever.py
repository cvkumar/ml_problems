import time

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf

from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

RANDOM_STATE = 2020

CATEGORIES = [
    "Special Occasion Makeup",
    "Twists",
    "Special Occasion Style",
    "Relaxers",
    "Massage",
    "Knots",
    "Locs",
    "Curling Treatments",
    "Braids",
    "Manicure",
    "Women's Haircut",
    "Natural Hair",
    "Hair Color",
    "Weaves",
    "Eyelashes",
    "Men's Color",
    "Sugaring",
    "Chemical Perms",
    "Pedicure",
    "Extensions",
    "Eyebrows",
    "Makeup",
    "Sets & Styles",
    "Style",
    "Barber",
    "Men's Haircut",
    "Wellness",
    "Hair Treatments",
    "Skincare",
    "Waxing",
    "Straightening Treatments",
    "Highlights",
]
CATEGORY_DICT = {
    "Special Occasion Makeup": {"predict_percent": 0.93},
    "Twists": {"predict_percent": 0.89},
    "Special Occasion Style": {"predict_percent": 0.83},
    "Relaxers": {"predict_percent": 0.70},
    "Massage": {"predict_percent": 0.90},
    "Knots": {"predict_percent": 0.95},
    "Locs": {"predict_percent": 0.7},
    "Curling Treatments": {"predict_percent": 0.99},
    "Braids": {"predict_percent": 0.57},
    "Manicure": {"predict_percent": 0.57},
    "Women's Haircut": {"predict_percent": 0.51},
    "Natural Hair": {"predict_percent": 0.72},
    "Hair Color": {"predict_percent": 0.52},
    "Weaves": {"predict_percent": 0.61},
    "Eyelashes": {"predict_percent": 0.61},
    "Men's Color": {"predict_percent": 0.93},
    "Sugaring": {"predict_percent": 0.94},
    "Chemical Perms": {"predict_percent": 0.82},
    "Pedicure": {"predict_percent": 0.65},
    "Extensions": {"predict_percent": 0.55},
    "Eyebrows": {"predict_percent": 0.51},
    "Makeup": {"predict_percent": 0.77},
    "Sets & Styles": {"predict_percent": 0.72},
    "Style": {"predict_percent": 0.58},
    "Barber": {"predict_percent": 0.57},
    "Men's Haircut": {"predict_percent": 0.57},
    "Wellness": {"predict_percent": 0.88},
    "Hair Treatments": {"predict_percent": 0.63},
    "Skincare": {"predict_percent": 0.71},
    "Waxing": {"predict_percent": 0.62},
    "Straightening Treatments": {"predict_percent": 0.77},
    "Highlights": {"predict_percent": 0.55},
}


def create_df_for_category(dataset: pd.DataFrame, category: str):
    results = {"service_name": [], "service_category": []}

    print(f"creating dataset for category: {category}")

    for index, row in dataset.iterrows():
        service_name = row["service_name"]
        entries = row["service_category"].split(",")
        entries = [entry.strip() for entry in entries]

        if category in entries:
            label = 1
        else:
            label = 0

        results["service_name"].append(service_name)
        results["service_category"].append(label)

    category_df = pd.DataFrame(
        data={
            "service_name": results["service_name"],
            "service_category": results["service_category"],
        }
    )
    print(category_df["service_category"].value_counts())
    df_groupby = category_df.groupby("service_category")
    category_df = df_groupby.apply(
        lambda x: x.sample(df_groupby.size().min()).reset_index(drop=True)
    )
    print(category_df["service_category"].value_counts())
    return category_df


def train_model_for_category(category_df, category, test_size=0.2):
    vocab_size = 10000
    embedding_dim = 16
    max_length = 50
    trunc_type = "post"
    padding_type = "post"
    oov_tok = "<OOV>"
    using_test_set = test_size > 0
    if not using_test_set:
        train_df = category_df
        test_df = pd.DataFrame(data={"service_name": [], "service_category": []})
    else:
        train_df, test_df = train_test_split(category_df, test_size=test_size)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    train_service_names = list(train_df["service_name"])
    train_labels = list(train_df["service_category"])
    test_service_names = list(test_df["service_name"])
    test_labels = list(test_df["service_category"])

    tokenizer.fit_on_texts(train_service_names)

    training_sequences = tokenizer.texts_to_sequences(train_service_names)
    training_padded = pad_sequences(
        training_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type,
    )

    testing_sequences = tokenizer.texts_to_sequences(test_service_names)
    testing_padded = pad_sequences(
        testing_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type,
    )

    training_padded = np.array(training_padded)
    training_labels = np.array(train_labels)

    testing_padded = np.array(testing_padded)
    testing_labels = np.array(test_labels)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, input_length=max_length
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    num_epochs = 10
    history = model.fit(
        training_padded,
        training_labels,
        epochs=num_epochs,
        validation_data=(testing_padded, testing_labels),
        verbose=0,
    )

    if using_test_set:
        test_predictions_raw = model.predict(testing_padded)
        test_predictions = [
            1 if prediction[0] > .5 else 0 for prediction in test_predictions_raw
        ]

        print(f"scores for category: {category}")
        print(
            metrics.classification_report(
                y_true=testing_labels, y_pred=test_predictions, digits=3
            )
        )

    return model, tokenizer


def get_count(df):
    categories = []
    for index, row in df.iterrows():
        entries = row['service_category'].split(',')
        entries = [entry.strip() for entry in entries]
        for entry in entries:
            categories.append(entry)

    counter = Counter(categories)
    print(counter)
    print(len(counter))


if __name__ == "__main__":
    start_time = time.time()
    vocab_size = 10000
    embedding_dim = 16
    max_length = 50
    trunc_type = "post"
    padding_type = "post"
    oov_tok = "<OOV>"

    df = pd.read_csv("data/styleseat/training_data.csv")
    train_df, test_df = train_test_split(df, test_size=0.2)

    get_count(df)

    category_to_model = {}
    category_to_tokenizer = {}

    test_service_names = list(test_df["service_name"])
    test_service_categories = list(test_df["service_category"])

    final_predictions = [[] for x in test_service_names]

    # categories = CATEGORIES[0:5]
    categories = CATEGORIES

    # TRAIN MODELS AND GET PREDICTIONS
    for category in categories:
        category_df = create_df_for_category(dataset=train_df, category=category)
        model, tokenizer = train_model_for_category(category_df, category, test_size=0)
        category_to_model[category] = model
        category_to_tokenizer[category] = tokenizer

        testing_sequences = tokenizer.texts_to_sequences(test_service_names)
        testing_padded = pad_sequences(
            testing_sequences,
            maxlen=max_length,
            padding=padding_type,
            truncating=trunc_type,
        )

        category_predictions_raw = model.predict(testing_padded)
        category_predictions = [
            1 if prediction[0] > CATEGORY_DICT[category]['predict_percent'] else 0 for prediction in category_predictions_raw
        ]

        for i in range(len(category_predictions)):
            prediction = category_predictions[i]
            if prediction == 1:
                final_predictions[i].append(category)

    for i in range(len(final_predictions)):
        final_predictions[i] = ", ".join(final_predictions[i])

    correct = 0
    partial_correct = 0
    total = 0
    incorrect = 0

    category_to_score = {category: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for category in categories}

    for i in range(len(final_predictions)):
        prediction = str(final_predictions[i])
        truth_category = test_service_categories[i]

        prediction_set = set([entry.strip() for entry in prediction.split(",")])
        truth_category_set = set([entry.strip() for entry in truth_category.split(",")])

        for category in categories:
            if category in prediction_set and category in truth_category_set:
                category_to_score[category]['tp'] = category_to_score[category]['tp'] + 1
            elif category in prediction_set and category not in truth_category_set:
                category_to_score[category]['fp'] = category_to_score[category]['fp'] + 1
            elif category not in prediction_set and category in truth_category_set:
                category_to_score[category]['fn'] = category_to_score[category]['fn'] + 1
            else:
                category_to_score[category]['tn'] = category_to_score[category]['tn'] + 1

        if prediction_set == truth_category_set:
            correct += 1
        else:
            incorrect += 1

        for prediction in prediction_set:
            if prediction in truth_category_set:
                partial_correct += 1
                break

        total += 1
    print(
        f"correct: {correct}, incorrect: {incorrect}, accuracy: {correct / (correct+incorrect)}"
    )
    print(
        f"partial correct: {partial_correct}, partial_accuracy: {partial_correct/total}"
    )

    print(f"Computing scores on a per category basis")

    for category, scores in category_to_score.items():
        try:
            precision = scores['tp'] / (scores['tp'] + scores['fp'])
        except Exception as e:
            precision = 0
            print("can't produce precision")

        try:
            recall = scores['tp'] / (scores['tp'] + scores['fn'])
        except Exception as e:
            recall = 0
            print("can't produce recall")

        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except Exception as e:
            f1 = 0
            print("can't produce f1")

        try:
            accuracy = (scores['tp'] + scores['tn']) / (scores['tp'] + scores['tn'] + scores['fp'] + scores['fn'])
        except Exception as e:
            accuracy = 0
            print("can't produce accuracy")

        print(f"Scores for category {category} - f1: {f1}, recall: {recall}, precision: {precision}, accuracy: {accuracy}")

    print("")
    print(f"program took: {time.time() - start_time} seconds")

"""
correct: 835, incorrect: 2765, accuracy: 0.23194444444444445

"""
