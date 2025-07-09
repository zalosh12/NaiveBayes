import pandas as pd


class Evaluator:
    def __init__(self,classifier):
        self.classifier = classifier

    def evaluate_model(self,X_test: pd.DataFrame,y_test: pd.Series):
        res = X_test.apply(self.classifier.predict,axis=1)
        correct = (res == y_test).sum()
        accuracy = correct / len(y_test) * 100

        return accuracy

        # print(len(X_test), len(y_test))
        # correct = 0
        # for i,row in X_test.iterrows():
        #     prediction = self.classifier.predict(row)
        #     if prediction == y_test.iloc[i]:
        #         correct += 1
        # return (correct / len(y_test)) * 100

    # def predict_single(self, sample_record) :
    #     return self.classifier.predict(sample_record)
