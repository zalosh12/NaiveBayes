import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
       self.model = {}
       self.class_priors = {}
       self.features = {}
       self.name = None

    def create_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.class_priors = y_train.value_counts(normalize=True).to_dict()
        self.features = {col: X_train[col].unique().tolist() for col in X_train.columns}
        # self.name = name
        labels = y_train.unique()

        for label in labels:
            self.model[label] = {}

            sub_X = X_train[y_train == label]

            for feature in self.features:
                value_count = sub_X[feature].value_counts()
                total = len(sub_X)

                possible_values = X_train[feature].unique()
                probs = {}

                for val in possible_values:
                    count = value_count.get(val,0)
                    probs[val] = (count + 1) / (total + len(possible_values))

                self.model[label][feature] = probs

    def predict(self,row: pd.Series):
        res = {}
        for label in self.model.keys():
            prob = self.class_priors[label]
            for feature in self.features:
                val_input = row[feature]
                prob *= self.model[label][feature].get(val_input,1e-6)
            res[label] = prob
        return max(res,key=res.get)







