from sklearn.neural_network import MLPClassifier

"""
For the neural network you should implement or steal your favorite kind of network and training algorithm. 
You may use networks of nodes with as many layers as you like and any activation function you see fit.
"""

X = [[0.0, 0.0], [1.0, 1.0]]
y = [0, 1]

clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)
print(clf.predict([[2.0, 2.0], [-1.0, -2.0]]))
