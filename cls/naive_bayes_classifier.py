from typing import Dict

import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
       self.model = None
       self.class_priors = None
       self.features = None

    def load_model(self,trained_model):
        self.model = trained_model["model"]
        self.class_priors = trained_model["class_priors"]
        self.features = trained_model["features"]

    def predict(self,sample_input:Dict) :
            res = {}
            for label in self.model.keys() :
                prob = self.class_priors[label]
                for feature in self.features :
                    val_input = sample_input[feature]
                    prob *= self.model[label][feature].get(val_input, 1e-6)
                res[label] = prob
            return max(res, key=res.get)