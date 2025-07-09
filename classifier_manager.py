from naive_bayes_classifier import NaiveBayesClassifier
from evaluator import Evaluator
from data_loader import LoadData
from data_spliter import DataSplitter
from user_interface import UserInterface
import pandas as pd

class ClassifierManager:
    def __init__(self) :
        self.df = None
        self.class_column = 'class'
        self.ui = UserInterface()
        self.classifier = NaiveBayesClassifier()
        self.evaluator = Evaluator(self.classifier)


    def run(self):
        # path = self.ui.enter_file_path()
        path = "phishing.csv"
        loader = LoadData(path)
        self.df = loader.data
        if self.df is None:
            return
        splitter = DataSplitter(self.df,class_column=self.class_column)
        X_train_df = splitter.X_train.copy()
        y_train_df = splitter.y_train
        X_test_df = splitter.X_test.copy()
        y_test_df = splitter.y_test

        self.classifier.create_model(X_train_df, y_train_df)

        mode = self.ui.choose_option()
        if mode == '1':
            accuracy = self.evaluator.evaluate_model(X_test_df,y_test_df)
            print(f'Accuracy: {accuracy:.2f}%')
        elif mode == '2':
            sample = self.ui.get_row_input(self.classifier.features)
            prediction = self.classifier.predict(sample)
            print(f"Predicted class: {prediction}")
        else:
            print('Invalid choice.')