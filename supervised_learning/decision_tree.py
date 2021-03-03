import sys

from sklearn import tree
import pandas as pd
import csv

"""
Decision Trees. For the decision tree, you should implement or steal a decision tree algorithm (and by "implement or steal" I mean "steal"). 
Be sure to use some form of pruning. You are not required to use information gain (for example, 
there is something called the GINI index that is sometimes used) to split attributes, but you should describe whatever it is that you do use.
"""
if __name__ == "__main__":

    df = pd.read_csv("census/census-income.csv")
    little_df = df[:10000]

    clf = tree.DecisionTreeRegressor()
    # clf = clf.fit(X, y)
    print(clf.predict([[1, 1]]))
