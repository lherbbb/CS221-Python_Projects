import numpy as np
###################################
# Optimization problem

trainExamples = [
    (1, 1),
    (2, 3),
    (4, 3)
]

def phi(x):
    return np.array([1, x])

def initialWeightVector():
    return np.zeros(2)

def trainLoss(w):
    return 1 / len(trainExamples) * sum((w.dot(phi(x)) - y)**2 for x, y in trainExamples)

def gradientTrainLoss(w):
    grad = np.zeros(2)
    for x, y in trainExamples:
        error = w.dot(phi(x)) - y
        grad += 2 * error * phi(x)
    return grad / len(trainExamples)

###################################
# Optimization Algorithm

def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.01
    for t in range(500):
        value = F(w)
        gradient = gradientF(w)
        w = w - eta * gradient
        print(f"Epoch {t}: w = {w}, F(w) = {value}, gradient = {gradient}")
    return w

# Run gradient descent
final_weights = gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)
