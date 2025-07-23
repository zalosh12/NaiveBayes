import pandas as pd


class NaiveBayesModel:
    def __init__(self):
        # Dictionary to hold the probabilities for each class and feature value
       self.model = {}
       self.class_priors = {}
       self.features = {}

    def build_model(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.class_priors = y_train.value_counts(normalize=True).to_dict()
        self.features = {col: x_train[col].unique().tolist() for col in x_train.columns}
        labels = y_train.unique()

        for label in labels:
            self.model[label] = {}

            sub_x = x_train[y_train == label]

            for feature in self.features:
                value_count = sub_x[feature].value_counts()
                total = len(sub_x)

                possible_values = x_train[feature].unique()
                probs = {}

                for val in possible_values:
                    count = value_count.get(val,0)
                    probs[val] = (count + 1) / (total + len(possible_values))

                self.model[label][feature] = probs