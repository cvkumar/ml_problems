import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow.python.keras.layers import Embedding

RANDOM_STATE = 2021
VALIDATION_SIZE = 0.2

df = pd.read_csv("data/titanic/train.csv")

nominal_categorical_variables = ["Sex"]

df = pd.get_dummies(df, columns=nominal_categorical_variables, drop_first=True)

X = pd.DataFrame(
    data={"age": df["Age"], "sex": df["Sex_male"], "p_class": df["Pclass"]}
)
Y = pd.DataFrame(data={"survived": df["Survived"]})

"""
x: age, sex, p_class
y: survived

"""

# le = preprocessing.LabelEncoder()
# X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, stratify=Y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
)

print(f"train: {x_train.shape}, test: {x_test.shape}")
model = tf.keras.models.Sequential(
    [
        # Embedding(input_dim=1, output_dim=1),
        # tf.keras.layers.Flatten(input_shape=(1, 1)),
        tf.keras.layers.Dense(32, activation="relu"),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

sex = np.array(x_train["sex"])
survived = np.array(y_train["survived"])

model.fit(sex, survived, epochs=6, verbose=2)
model.summary()

print("evaluation")
model.evaluate(x_test["sex"], y_test["survived"])

# PRECISION, F1, RECALL
predictions = model.predict(x_test["sex"])
labels = np.array(y_test["survived"])
false_negatives, false_positives, true_positives, true_negatives = 0, 0, 0, 0
for i in range(len(predictions)):
    prediction = np.argmax(predictions[i])
    label = labels[i]

    if prediction == 1 and prediction == label:
        true_positives += 1
    elif prediction == 0 and prediction == label:
        true_negatives += 1
    elif prediction == 0 and label == 1:
        false_negatives += 1
    elif prediction == 1 and label == 0:
        false_positives += 1
    else:
        print("THIS SHOULD NEVER HAPPEN")

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = (2 * precision * recall) / (precision + recall)

print(f"precision: {precision}, recall: {recall}, f1: {f1}")
