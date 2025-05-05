from collections import defaultdict, Counter
import pandas as pd
import os, json
import numpy as np

class NaiveBayesClass:
    def __init__(self, X=None, Y=None, type="discreate"):
        self.X = X
        self.Y = Y
        self.type = type

        if X != None:
            self.dimension = self.get_dimensions(self.Y)
            self.unique_X = list(set([j for i in self.X for j in i]))
            if self.dimension == 1:
                self.unique_Y = list(set(self.Y))
            elif self.dimension == 2:
                self.unique_Y = list(set(y for sublist in Y for y in sublist))
            self.prior_probability()
            self.posterior_probability()

    def get_dimensions(self, arr):
        if isinstance(arr, list):
            if isinstance(arr[0], list):
                return 2
            else:
                return 1
        else:
            raise ValueError("Y must be a list")
    
    def prior_probability(self):
        
        prior = defaultdict(float)
        if self.dimension == 1:
            denominator = len(self.Y)
            nominator = Counter(self.Y)        
        elif self.dimension == 2:
            denominator = sum(len(y) for y in self.Y)
            flattened_X = [y for sublist in self.Y for y in sublist]
            nominator = Counter(flattened_X)
        else:
            raise ValueError("Supported Y dimensions only single class label (1D) or sequence class label (2D)")

        for key, val in nominator.items():
            prior[key] = (val + 1) / (denominator + + len(self.unique_Y))
        
        self.prior = prior
    
    def posterior_probability(self):
        posterior = defaultdict(lambda: defaultdict(float))
        if self.type == "discreate":
            nominator = defaultdict(Counter)
            denominator = defaultdict(int)

            for i in range(len(self.X)):
                for j in range(len(self.X[i])):
                    if self.dimension == 1:
                        nominator[self.Y[i]][self.X[i][j]] += 1
                    elif self.dimension == 2:
                        nominator[self.Y[i][j]][self.X[i][j]] += 1
                        denominator[self.Y[i][j]] += 1
                if self.dimension == 1:
                    denominator[self.Y[i]] += len(self.X[i])
                
            for y in self.unique_Y:
                for x in self.unique_X:
                    posterior[y][x] = (nominator[y][x] + 1) / (denominator[y] + len(self.unique_X))  # Laplace smoothing
            
            self.posterior = posterior

        elif self.type == "continous":
            mu = defaultdict(lambda: defaultdict(float))
            variance = defaultdict(lambda: defaultdict(float))
            denominator = Counter(self.Y)

            for i in range(len(self.X[0])):
                for k in self.unique_Y:
                    nominator = sum(self.X[j][i] for j in range(len(self.Y)) if self.Y[j] == k)
                    mu[i][k] = nominator / denominator[k]

            for i in range(len(self.X[0])):
                for k in self.unique_Y:
                    nominator = sum((self.X[j][i] - mu[i][k]) ** 2  for j in range(len(self.Y)) if self.Y[j] == k)
                    variance[i][k] = nominator / (denominator[k]-1)
            self.mu = mu
            self.variance = variance          

    def predict(self, X, type=None):
            if type is None:
                type = self.type

            if type == "discreate":
                max_prob = -float('inf')
                label_class = None
                for y in self.prior:
                    prob = np.log(self.prior[y])
                    for i in X:
                        prob += np.log(self.posterior[y].get(i, 1e-6))
                    if prob > max_prob:
                        max_prob = prob
                        label_class = y
            elif type == "continous":
                max_prob = -float('inf')
                label_class = None
                for k in self.prior.keys():
                    prob = np.log(self.prior[k])
                    for i in range(len(X)):
                        mu = self.mu[str(i)][k]
                        variance = self.variance[str(i)][k]

                        if variance > 0:
                            prob += -((X[i] - mu)**2) / (2 * variance) - 0.5 * np.log(2 * np.pi * variance)
                        else:
                            prob = -float('inf')
                        # print(prob, k, max_prob, prob > max_prob, variance)
                    if prob > max_prob:
                        max_prob = prob
                        label_class = k
                        # print("after condition",max_prob, k)

            return label_class


    def save_parameters(self, filepath="params.json"):
        """
        this function is used for saving the parameter model into .json format
        """
        if len(filepath.split(".")) < 2 or 'json' not in filepath.split("."):
            raise ValueError("Filepath must be in .json format")
        os.makedirs("model", exist_ok=True)
        if self.type == "discreate":
            data = {
                "prior" : dict(self.prior),
                "posterior" : {k: dict(v) for k, v in self.posterior.items()},
            }
        elif self.type == "continous":
                data = {
                "prior" : dict(self.prior),
                "mu" : {k: dict(v) for k, v in self.mu.items()},
                "variance" : {k: dict(v) for k, v in self.variance.items()}
            }
        with open(f"model/{filepath}", "w") as f:
            json.dump(data, f, indent=4)
            print(f"Saved model in model/{filepath}")
    
    def load_parameters(self, filepath="params.json"):
        with open(filepath, "r") as f:
            data = json.load(f) # (library: json)

        json_keys = list(data.keys())

        for key in json_keys:
            if "posterior" == key:
                self.posterior = {k: defaultdict(float, v) for k, v in data[key].items()} #(library: collections)
            elif "mu" == key:
                self.mu = {k: defaultdict(float, v) for k, v in data[key].items()} #(library: collections)
            elif "variance" == key:
                self.variance = {k: defaultdict(float, v) for k, v in data[key].items()} #(library: collections)
            else:
                self.prior = defaultdict(float, data[key]) #(library: collections)

    def accuracy(self, X, Y, verbose=True):
        """
        AccuracY = true label recognized /  all label sequence
        """

        states = list(set([s for seq in Y for s in seq]))
        if verbose:
            print(f"\t{'label':<10} | {'accuracY'}\n")
        tp_all = 0
        total_all = sum([len(seq) for seq in Y])
        
        for s in states:
            tp = 0
            total = 0
            for i in range(len(X)):
                Y_pred = self.predict(X[i])
                for j in range(len(Y_pred)):
                    if Y[i][j] == s:
                        if Y_pred[j] == Y[i][j]:
                            tp += 1
                        total += 1
            if verbose:
                print(f"\t{s:<10} | {tp/total:.2f}")
        
        for i in range(len(X)):
            Y_pred = self.predict(X[i])
            for j in range(len(Y_pred)):
                if Y_pred[j] == Y[i][j]:
                    tp_all += 1
        if verbose:
            print(f"\n{'Accuracy total':>10} | {tp_all/total_all:.2f}")
        return tp_all/total_all
        