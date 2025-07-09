from naive_bayes_classifier import NaiveBayesClassifier
from evaluator import Evaluator
from data_loader import LoadData
from data_spliter import DataSplitter
from user_interface import UserInterface

loader = LoadData("phishing.csv")
splitter = DataSplitter(loader.data)
print(splitter.X_test.head())
print(splitter.y_test.head())