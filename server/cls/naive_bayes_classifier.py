import pandas as pd

class NaiveBayesClassifier:
    def __init__(self,trained_model):
       self.model = trained_model.model
       self.class_priors = trained_model.class_priors
       self.features = trained_model.features


    def predict(self,row: pd.Series) :
            res = {}
            for label in self.model.keys() :
                prob = self.class_priors[label]
                for feature in self.features :
                    val_input = row[feature]
                    prob *= self.model[label][feature].get(val_input, 1e-6)
                res[label] = prob
            return max(res, key=res.get)