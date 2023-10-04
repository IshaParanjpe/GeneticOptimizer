import numpy as np
from sklearn.metrics import mean_squared_error
import random
import warnings
warnings.filterwarnings('ignore')

random.seed(12)
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape


        def func(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)
    

        def fitness(y, s, b):
            y_predicted = np.dot(X,np.array(s)) + b
            return func(y,y_predicted)

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.plot_arr = []

        # genetic trial
        self.solutions = []
        for i in range(1000):
            temp = []
            for _ in range(n_features + 1):
                temp.append(random.uniform(0, 10000))

            self.solutions.append(tuple(temp))
            
        for i in range(1000):
            rankedSolutions = []
            # Calculate Fitness
            for s in self.solutions:
                rankedSolutions.append((fitness(y,s[:-1],s[-1]), (s)))
                
            rankedSolutions.sort(key = lambda x: x[0])
            print(f"------- Gen {i} Best Solutions -------")
            print(rankedSolutions[0])

            # stopping criterion
            if rankedSolutions[0][0] < 5:
                print("Mila bhai")
                break
            
            # Selection
            # otherwise (i.e. if the stopping criteria is not satisfied)
            # select the top 100 solutions
            bestSolutions = rankedSolutions[:100]

            self.plot_arr.append(bestSolutions[0][0])


            # s = (x,y,z)
            elements = []
            for s in bestSolutions:
                elements.append(s[1][0]) #s
                elements.append(s[1][1]) #s
                elements.append(s[1][2]) #s


            # crossover and mutation
            newGen = []
            for _ in range(1000):
    
                e1 = np.array(random.choice(elements)) * random.uniform(0.99, 1.01)
                e2 = np.array(random.choice(elements)) * random.uniform(0.99, 1.01)
                e3 = np.array(random.choice(elements)) * random.uniform(0.99, 1.01)

                newGen.append((e1,e2,e3))

            self.solutions = newGen

        self.weights, self.bias = self.solutions[0][0], self.solutions[0][1]

        return self.plot_arr



    def predict(self, X):
        best_sol = self.solutions[0]
        y_approx = np.dot(X, np.array(best_sol[:-1])) + best_sol[-1]
        return np.array(y_approx)

random.seed(42)
class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape


        def func(y_true, y_pred):
            y_zero_loss = y_true * np.log(y_pred + 1e-9)
            y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
            return -np.mean(y_zero_loss + y_one_loss)
    

        def fitness(y, s, b):
            linear_model = np.dot(X,np.array(s)) + b
            y_predicted = self._sigmoid(linear_model)
            # print('predicted stuff \n', np.where(y_predicted > 0.5, 1 , 0))
            return func(y.reshape(1,X.shape[0]),y_predicted)

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.plot_arr = []

        # genetic trial
        self.solutions = []
        for i in range(1000):
            temp = []
            for _ in range(n_features + 1):
                temp.append(random.uniform(0, 10000))

            self.solutions.append(tuple(temp))
            
        for i in range(1000):
            rankedSolutions = []
            # Calculate Fitness
            for s in self.solutions:
                rankedSolutions.append((fitness(y,s[:-1],s[-1]), (s)))
                
            rankedSolutions.sort(key = lambda x: x[0])
            print(f"------- Gen {i} Best Solutions -------")
            print(rankedSolutions[0])

            # stopping criterion
            if rankedSolutions[0][0] < 0.5:
                print("Got the result")
                break
            
            # Selection
            # otherwise (i.e. if the stopping criteria is not satisfied)
            # select the top 100 solutions
            bestSolutions = rankedSolutions[:100]

            # maintaning a record of the loss function
            self.plot_arr.append(bestSolutions[0][0])


            # s = (x,y,z)
            elements = []
            for s in bestSolutions:
                elements.append(s[1][0]) # weights
                elements.append(s[1][1]) # weights
                elements.append(s[1][2]) # bias


            # crossover and mutation
            newGen = []
            for _ in range(1000):
    
                e1 = np.array(random.choice(elements)) * random.uniform(0.99, 1.01)
                e2 = np.array(random.choice(elements)) * random.uniform(0.99, 1.01)
                e3 = np.array(random.choice(elements)) * random.uniform(0.99, 1.01)

                newGen.append((e1,e2,e3))

            self.solutions = newGen

        return self.plot_arr


    def predict(self, X):
        best_sol = self.solutions[0]
        linear_model = np.dot(X, np.array(best_sol[:-1])) + best_sol[-1]
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def accuracy(self,y, y_hat):
        accuracy = np.sum(np.equal(y,y_hat))/len(y)
        return accuracy