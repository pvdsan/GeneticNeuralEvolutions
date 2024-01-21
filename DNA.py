import random
import math
import numpy as np
from sklearn.datasets import *
from sklearn.neural_network import MLPClassifier

# Define the arrays for activation functions and solvers
activationFunc = ['tanh', 'identity', 'logistic', 'relu']
solverArray = ['adam', 'sgd', 'lbfgs']

def get_dimensions():
    rand_num = random.randint(1, 6)
    random_array = list(np.random.rand(rand_num))
    for i in range(0, len(random_array)):
        random_array[i] = random_array[i]*1000
        random_array[i] = math.floor(random_array[i])
    return random_array

def get_activation():
    rand_num = random.randint(0, 3)
    return activationFunc[rand_num]

def get_solver():
    rand_num = random.randint(0, 2)
    return solverArray[rand_num]

class DNA:
    
    def __init__(self):
        self.max_iter = random.randint(100, 200)
        self.learning_rate_init = round(random.uniform(0.1, 0.5), 2)
        self.dimensions = get_dimensions()
        self.fitness = 0
        self.solver = get_solver()
        self.activation = get_activation()
        self.mlp = MLPClassifier()
        self.X_train = []
        self.y_train = []
        self.TrainingACC = 0
        self.TestingACC = 0
        self.X_test = []
        self.y_test = []

    def crossover(self, partner):
        child = DNA()  # Assumes DNA() is a constructor for a new DNA object.

        p1 = len(self.dimensions)
        p2 = len(partner.dimensions)
        
        if p1 > p2:
            child_layers = random.randint(p2, p1)
        else:
            child_layers = random.randint(p1, p2)
        
        child.dimensions = []
        
        for i in range(0, child_layers):
            x = random.randint(100, 512)
            child.dimensions.append(x)
        
        child.learning_rate_init = (self.learning_rate_init + partner.learning_rate_init) / 2
        child.max_iter = int((self.max_iter + partner.max_iter) / 2)
        
        solver_mate_pool = [self.solver, partner.solver]
        activation_mate_pool = [self.activation, partner.activation]
        
        child.solver = random.choice(solver_mate_pool)
        child.activation = random.choice(activation_mate_pool)
        
        return child

    def setNN(self):
        self.mlp = MLPClassifier(hidden_layer_sizes=tuple(self.dimensions), max_iter=self.max_iter, learning_rate_init=self.learning_rate_init)

    def CreateSamples(self):
        X, y = load_digits(return_X_y=True)
        X = X / 255.
        
        # rescale the data, use the traditional train/test split
        self.X_train, self.X_test = X[:1200], X[1200:]
        self.y_train, self.y_test = y[:1200], y[1200:]

    def calcFitness(self):
        self.mlp.fit(self.X_train, self.y_train)
        
        self.TrainingACC = self.mlp.score(self.X_train, self.y_train)
        self.TestingACC = self.mlp.score(self.X_test, self.y_test)
        
        self.fitness = self.TrainingACC + self.TestingACC