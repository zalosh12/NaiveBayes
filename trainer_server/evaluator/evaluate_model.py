import pandas as pd

class Evaluate:
    """Evaluate a model by calculating accuracy and making predictions."""
    def __init__(self,model_to_evaluate):
        self.model_to_evaluate = model_to_evaluate

    def evaluate_model(self,x_test: pd.DataFrame,y_test: pd.Series) -> float:
        """Calculate accuracy on test data."""
        correct = 0
        for i, row in x_test.iterrows() :
            prediction = self.predict(row)
            if prediction == y_test.loc[i]:
                correct += 1
        accuracy = correct / len(y_test)

        return float(accuracy)

    def predict(self, row: pd.Series) :
            """Predict the class for one data sample."""
            res = {}
            for label in self.model_to_evaluate.model.keys() :
                prob = self.model_to_evaluate.class_priors[label]
                for feature in self.model_to_evaluate.features :
                    val_input = row[feature]
                    prob *= self.model_to_evaluate.model[label][feature].get(val_input, 1e-6)
                res[label] = prob
            return max(res, key=res.get)