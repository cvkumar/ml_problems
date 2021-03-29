import random

"""
Train data that inputs 3 numbers 0-9. 

When the sum of those numbers is greater than 12, output 1
otherwise output 0 
"""
RANDOM_STATE = 12
random.seed(RANDOM_STATE)

INPUT_DATA_LENGTH = 10000

X1 = [random.randint(1, 9) for num in range(INPUT_DATA_LENGTH)]
X2 = [random.randint(1, 9) for num in range(INPUT_DATA_LENGTH)]
X3 = [random.randint(1, 9) for num in range(INPUT_DATA_LENGTH)]

Y = []
for i in range(INPUT_DATA_LENGTH):
    Y.append(X1[i] + X2[i] + X3[i])
    # if X1[i] + X2[i] + X3[i] > 13.5:
    #     Y.append(1)
    # else:
    #     Y.append(0)


# print(X)
# print(W)
# print(Z)
# print(Y)


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w1 = random.random() * 1
        self.w2 = random.random() * 1
        self.w3 = random.random() * 1

    def train(self, x1, x2, x3, y):
        for i in range(INPUT_DATA_LENGTH):

            if i % 50 == 0:
                print(
                    f"at step i: {i} weight1: {self.w1}, weight2: {self.w2}, weight3: {self.w3}"
                )

            output = self._compute_output(x1[i], x2[i], x3[i])

            self.w1 = self.w1 + (self.learning_rate * (y[i] - output) * x1[i])
            self.w2 = self.w2 + (self.learning_rate * (y[i] - output) * x2[i])
            self.w3 = self.w3 + (self.learning_rate * (y[i] - output) * x3[i])

    def eval(self, x1, x2, x3, y):
        correct = 0
        for i in range(INPUT_DATA_LENGTH):
            output = self._compute_output(x1[i], x2[i], x3[i])
            if output == y[i]:
                correct += 1

        print(f"Accuracy: {correct / INPUT_DATA_LENGTH}")
        return

    def predict(self, a: int, b: int, c: int):
        # return self.w1 * a + self.w2 * b + self.w3 * c
        output = self._compute_output(a, b, c)
        return output

    def _compute_output(self, a, b, c):
        output = self.w1 * a + self.w2 * b + self.w3 * c

        # Is this the activation function?
        # if output > 0:
        #     output = 1
        # else:
        #     output = 0
        return output


if __name__ == "__main__":
    network = NeuralNetwork()

    network.train(X1, X2, X3, Y)

    # print(network.predict(10, 10, 10))

    print("EVAL TIME:")

    network.eval(X1, X2, X3, Y)
