import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Embedding

RANDOM_STATE = 2021
VALIDATION_SIZE = 0.2
AVERAGE_AGE = 24

df = pd.read_csv("data/titanic/train.csv")

nominal_categorical_variables = ["Sex", "Embarked"]

df = pd.get_dummies(df, columns=nominal_categorical_variables, drop_first=True)

X = pd.DataFrame(
    data={
        "age": df["Age"],
        "sex": df["Sex_male"],
        "p_class": df["Pclass"],
        "fare": df["Fare"],
        "embarked_q": df['Embarked_Q'],
        "embarked_s": df['Embarked_S'],
    }
)
Y = pd.DataFrame(data={"survived": df["Survived"]})

X["age"].fillna(AVERAGE_AGE, inplace=True)

# TODO: Add age group, embarked
# TODO: Add method to be able to run simple regression on one input

sex_input = keras.Input(shape=(1,), name="sex")
# age_input = keras.Input(shape=(1,), name="age")
p_class_input = keras.Input(shape=(1,), name="p_class")
fare_input = keras.Input(shape=(1,), name="fare")
embarked_q = keras.Input(shape=(1,), name="embarked_q")
embarked_s = keras.Input(shape=(1,), name="embarked_s")

x = layers.concatenate([sex_input, p_class_input, fare_input, embarked_q, embarked_s])

# Stick a logistic regression for priority prediction on top of the features
# hidden_layer = layers.Dense(64, name="hidden", activation="relu")(x)
survived_prediction = layers.Dense(2, name="survived", activation="softmax")(x)

model = keras.Model(
    inputs=[sex_input, p_class_input, fare_input, embarked_s, embarked_q], outputs=[survived_prediction],
)

# model.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss={"survived": keras.losses.BinaryCrossentropy(from_logits=True)},
#     metrics=[keras.metrics.BinaryAccuracy()],
#     # loss_weights=[1.0, 0.2],
# )

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])


# model.summary()
# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, stratify=Y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
)

# model.fit(
#     {"sex": x_train["sex"], "p_class": x_train["p_class"], "fare": x_train["fare"], "embarked_q": x_train['embarked_q'], "embarked_s": x_train["embarked_s"]},
#     {"survived": y_train["survived"]},
#     epochs=200,
#     # batch_size=16,
# )

print("evaluate")

model.evaluate(
    {"sex": x_test["sex"], "p_class": x_test["p_class"], "fare": x_test["fare"], "embarked_q": x_test['embarked_q'], "embarked_s": x_test["embarked_s"]},
    {"survived": y_test["survived"]},
)

print("")


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(x_train, y_train)
Y_prediction = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

"""
sources:

https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
- What inputs matter


"""
# TODO: https://www.kaggle.com/theblackmamba31/titanic-tutorial-neural-network
